"""
dysarthria_analysis.py
----------------------
Acoustic analysis pipeline for tracking FLACCID DYSARTHRIA ("disartria flácida")
recovery after stroke.

It maps raw acoustic measures to five clinical domains:
    respiratory_support   <- phonation_duration_sec (max phonation time) [+ phrase length]
    voice_stability       <- jitter, shimmer, HNR, volume_std_db
    labial_closure        <- bilabial modulation depth / burst sharpness in /pa/ DDK
    ddk_regular           <- ddk_cv_interval (+ ddk_rate)
    intelligibility       <- word accuracy vs. the known target text (optional)

Recording protocol (record as SEPARATE files / segments):
    1. Sustained vowel   : patient sustains /a/ as long & steady as possible (1 try, best of 3)
    2. DDK                : fast repetition of /pa/ (or /pa-ta-ka/) for ~5-8 s
    3. Reading passage    : the standard lines you give them to read

This is a SCREENING / TRACKING aid, not a diagnosis. The numbers complement,
they do not replace, the speech-language pathologist's perceptual judgment.
"""

from __future__ import annotations
import json
import os
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call


# ----------------------------------------------------------------------------
# Reference values (ADJUSTABLE defaults from common literature / Praat-MDVP).
# Calibrate these to your own clinic and the patient's age/sex if you can.
# ----------------------------------------------------------------------------
NORMS = {
    "phonation_duration_sec": {"good": 15.0, "poor": 6.0},   # max phonation time
    "jitter_local_pct":       {"good": 1.04, "poor": 3.0},   # lower better
    "shimmer_local_pct":      {"good": 3.81, "poor": 10.0},  # lower better
    "hnr_db":                 {"good": 20.0, "poor": 7.0},   # higher better
    "volume_std_db_sustain":  {"good": 1.5,  "poor": 6.0},   # lower better (sustained /a/)
    "ddk_rate_syll_sec":      {"good": 6.0,  "poor": 3.0},   # higher better (/pa/)
    "ddk_cv_interval":        {"good": 0.10, "poor": 0.35},  # lower better (more regular)
    "labial_mod_depth":       {"good": 0.85, "poor": 0.35},  # higher better (lip closure, /pa/)
    "lingual_mod_depth":      {"good": 0.85, "poor": 0.35},  # higher better (tongue closure, /ta/ /ka/)
    "smr_rate_syll_sec":      {"good": 5.0,  "poor": 2.5},   # higher better (/pataka/ sequence)
    "intelligibility_pct":    {"good": 95.0, "poor": 50.0},
}

# Canonical place-of-articulation order by burst spectral centroid (ascending):
#   /p/ bilabial  -> lowest  (diffuse, low-frequency, weak burst)
#   /k/ velar     -> middle  (compact mid-frequency burst)
#   /t/ alveolar  -> highest (sharp high-frequency burst)
PATAKA_ORDER = ["pa", "ka", "ta"]


def _score(value, good, poor):
    """Map a raw value onto 0-100 where 100 = at/above 'good', 0 = at/below 'poor'.
    Handles both 'higher is better' (good>poor) and 'lower is better' (good<poor)."""
    if value is None or np.isnan(value):
        return None
    if good >= poor:  # higher is better
        s = (value - poor) / (good - poor)
    else:             # lower is better
        s = (poor - value) / (poor - good)
    return round(float(max(0.0, min(1.0, s))) * 100, 1)


# ----------------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------------
_FFMPEG = shutil.which("ffmpeg")


def ensure_pcm_wav(path, sr=44100, cache_dir=None, force=False):
    """Re-encode any audio file (m4a, AAC, float/odd WAV, ...) to mono 16-bit PCM
    WAV via ffmpeg, so Praat/parselmouth can read it reliably. Returns the path to
    the converted file. Results are cached, so repeat runs don't re-encode.

    If ffmpeg is not installed, prints a warning and returns the original path
    unchanged (the caller's loader fallback may still cope with plain WAVs).
    """
    if path is None:
        return None
    if _FFMPEG is None:
        print("[warn] ffmpeg not found on PATH; using original file as-is. "
              "Install it for robust decoding:  sudo apt install ffmpeg")
        return path

    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "dysarthria_wav")
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(cache_dir, f"{base}__{sr}hz_mono.wav")

    # skip if a fresh cached copy already exists
    if (not force and os.path.exists(out)
            and os.path.getmtime(out) >= os.path.getmtime(path)):
        return out

    try:
        subprocess.run(
            [_FFMPEG, "-y", "-i", path, "-ac", "1", "-ar", str(sr),
             "-c:a", "pcm_s16le", out],
            check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip().splitlines()
        print(f"[warn] ffmpeg could not convert {path}: "
              f"{tail[-1] if tail else 'unknown error'}")
        return path
    return out


def load_sound(path, target_sr=44100):
    try:
        return parselmouth.Sound(path)          # fast path: Praat reads it directly
    except parselmouth.PraatError:
        try:
            import soundfile as sf
            y, sr = sf.read(path, always_2d=False)   # handles float/odd WAVs
        except Exception:
            y, sr = librosa.load(path, sr=None, mono=True)  # last resort (uses ffmpeg)
        y = np.asarray(y, dtype="float64")
        if y.ndim > 1:                          # stereo -> mono
            y = y.mean(axis=1)
        return parselmouth.Sound(y, sampling_frequency=sr)


def _voiced_intensity_std(snd, pitch=None):
    """Std of the intensity contour (dB) over voiced frames -> volume_std_db."""
    intensity = snd.to_intensity(minimum_pitch=75)
    vals = intensity.values[0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    # keep only frames within 35 dB of the peak (the actual phonation, not silence)
    vals = vals[vals > (vals.max() - 35)]
    return float(np.std(vals)) if vals.size else float("nan")


# ----------------------------------------------------------------------------
# 1. SUSTAINED VOWEL  ->  respiratory_support + voice_stability
# ----------------------------------------------------------------------------
def analyze_sustained_vowel(path, f0min=75, f0max=400):
    snd = load_sound(path)
    pitch = snd.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max)

    # --- phonation duration: longest continuous voiced run -------------------
    f0 = pitch.selected_array["frequency"]      # 0 where unvoiced
    dt = pitch.time_step
    voiced = f0 > 0
    best = run = 0
    for v in voiced:
        run = run + 1 if v else 0
        best = max(best, run)
    phonation_duration_sec = best * dt

    # --- perturbation: jitter / shimmer / HNR --------------------------------
    pp = call([snd, pitch], "To PointProcess (cc)")
    jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100      # %
    shimmer = call([snd, pp], "Get shimmer (local)",
                   0, 0, 0.0001, 0.02, 1.3, 1.6) * 100                          # %
    harm = snd.to_harmonicity_cc(minimum_pitch=f0min)
    hv = harm.values[harm.values != -200]
    hnr = float(hv.mean()) if hv.size else float("nan")

    volume_std_db = _voiced_intensity_std(snd, pitch)

    return {
        "phonation_duration_sec": round(float(phonation_duration_sec), 2),
        "jitter_local_pct":       round(float(jitter), 3),
        "shimmer_local_pct":      round(float(shimmer), 3),
        "hnr_db":                 round(hnr, 2),
        "volume_std_db":          round(float(volume_std_db), 2),
    }


# ----------------------------------------------------------------------------
# Shared onset detection + burst spectral features (used by AMR and SMR tasks)
# ----------------------------------------------------------------------------
def _detect_onsets(y, sr, hop=256, sensitivity=0.6):
    """Return (onset_times, peak_frame_indices, rms_envelope, hop)."""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.times_like(env, sr=sr, hop_length=hop)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5,
                                   post_avg=5, delta=np.median(env) * sensitivity, wait=4)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    return times[peaks], peaks, rms, hop


def _burst_features(y, sr, onset_time, win_ms=35):
    """Spectral centroid (Hz) and high-frequency energy ratio of the burst
    window just after an onset. Discriminates place of articulation:
    /p/ low centroid, /k/ mid, /t/ high."""
    start = int(onset_time * sr)
    n = int(sr * win_ms / 1000)
    seg = y[start:start + n]
    if len(seg) < 16:
        return None
    win = np.hanning(len(seg))
    S = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    total = S.sum() + 1e-9
    centroid = float((freqs * S).sum() / total)
    hf_ratio = float(S[freqs > 2000].sum() / total)
    return centroid, hf_ratio


def _valley_depth(rms, a, b):
    """Closure depth between two peak frames: (peak - valley)/peak. 1 = full stop."""
    if b - a < 2:
        return None
    seg = rms[a:b]
    peak_amp = max(rms[a], rms[b]) + 1e-9
    return float((peak_amp - seg.min()) / peak_amp)


def _classify_pataka(features):
    """Assign each onset to pa/ta/ka by clustering burst centroids into 3 groups,
    then labelling the clusters by ascending centroid (pa<ka<ta). Robust to a
    dropped/extra syllable because it does NOT rely on strict cyclic position."""
    from scipy.cluster.vq import kmeans2
    cents = np.array([f[0] for f in features], dtype=float)
    hf = np.array([f[1] for f in features], dtype=float)
    # standardize the 2 features
    X = np.column_stack([
        (cents - cents.mean()) / (cents.std() + 1e-9),
        (hf - hf.mean()) / (hf.std() + 1e-9),
    ])
    _, labels = kmeans2(X, 3, minit="++", seed=0, missing="warn")
    # order clusters by their mean centroid -> lowest=pa, mid=ka, highest=ta
    order = sorted(range(3), key=lambda c: cents[labels == c].mean()
                   if np.any(labels == c) else np.inf)
    cluster_to_place = {order[0]: "pa", order[1]: "ka", order[2]: "ta"}
    return [cluster_to_place[l] for l in labels]


# ----------------------------------------------------------------------------
# 2b. /PATAKA/ SMR -> labial_closure + lingual_closure + per-place breakdown
# ----------------------------------------------------------------------------
def analyze_pataka(path):
    """Sequential motion rate (SMR). Separates the three places of articulation
    so labial (/pa/, facial nerve) and lingual (/ta/ /ka/, hypoglossal) can be
    scored independently -- the localization payoff of /pataka/."""
    y, sr = librosa.load(path, sr=None, mono=True)
    onsets, peaks, rms, hop = _detect_onsets(y, sr)

    result = {"smr_n_syllables": int(len(onsets))}
    if len(onsets) < 6:   # need ~2 full cycles to be meaningful
        for k in ("smr_rate_syll_sec", "smr_cv_interval",
                  "labial_mod_depth", "lingual_mod_depth"):
            result[k] = float("nan")
        result["per_place"] = {}
        return result

    intervals = np.diff(onsets)
    mean_iv = float(np.mean(intervals))
    result["smr_rate_syll_sec"] = round(1.0 / mean_iv, 2) if mean_iv > 0 else float("nan")
    result["smr_cv_interval"] = round(float(np.std(intervals) / mean_iv), 3) if mean_iv > 0 else float("nan")

    # classify each onset by place
    feats = [_burst_features(y, sr, t) for t in onsets]
    valid = [i for i, f in enumerate(feats) if f is not None]
    places = _classify_pataka([feats[i] for i in valid])
    place_of = {valid[j]: places[j] for j in range(len(valid))}

    # per-place metrics: closure depth, burst centroid, and inter-onset
    # regularity measured at every occurrence of that place (= cycle period)
    per_place = {}
    for p in PATAKA_ORDER:
        idxs = [i for i in valid if place_of[i] == p]
        depths, cents = [], []
        for i in idxs:
            cents.append(feats[i][0])
            if i + 1 < len(peaks):
                d = _valley_depth(rms, peaks[i], peaks[i + 1])
                if d is not None:
                    depths.append(d)
        # regularity of this place across the recording
        ptimes = sorted(onsets[i] for i in idxs)
        place_cv = float("nan")
        if len(ptimes) >= 3:
            piv = np.diff(ptimes)
            place_cv = round(float(np.std(piv) / np.mean(piv)), 3) if np.mean(piv) > 0 else float("nan")
        per_place[p] = {
            "n": len(idxs),
            "mod_depth": round(float(np.mean(depths)), 3) if depths else float("nan"),
            "centroid_hz": round(float(np.mean(cents)), 1) if cents else float("nan"),
            "cv_interval": place_cv,
        }
    result["per_place"] = per_place

    # roll up into the closure domains
    result["labial_mod_depth"] = per_place["pa"]["mod_depth"]
    lingual = [per_place["ta"]["mod_depth"], per_place["ka"]["mod_depth"]]
    lingual = [x for x in lingual if not np.isnan(x)]
    result["lingual_mod_depth"] = round(float(np.mean(lingual)), 3) if lingual else float("nan")
    return result


# ----------------------------------------------------------------------------
# 2. DDK  ->  ddk_regular + labial_closure
# ----------------------------------------------------------------------------
def analyze_ddk(path, syllable="pa"):
    """Detects syllable onsets in a /pa/ (alternating motion rate, AMR) task."""
    y, sr = librosa.load(path, sr=None, mono=True)
    onsets, peaks, rms, hop = _detect_onsets(y, sr)

    result = {"ddk_n_syllables": int(len(onsets))}
    if len(onsets) >= 3:
        intervals = np.diff(onsets)
        # trim 1st/last in case of recording edges
        mean_iv = float(np.mean(intervals))
        cv = float(np.std(intervals) / mean_iv) if mean_iv > 0 else float("nan")
        result["ddk_rate_syll_sec"] = round(1.0 / mean_iv, 2)
        result["ddk_cv_interval"] = round(cv, 3)
    else:
        result["ddk_rate_syll_sec"] = float("nan")
        result["ddk_cv_interval"] = float("nan")

    # --- labial closure proxy (only meaningful for bilabial /pa/) ------------
    # Full lip closure -> deep amplitude valleys between syllables (true stop).
    # Weak closure -> shallow valleys (lips never fully seal).
    if len(onsets) >= 2 and "p" in syllable:
        depths = []
        for i in range(len(peaks) - 1):
            a, b = peaks[i], peaks[i + 1]
            if b - a < 2:
                continue
            seg = rms[a:b]
            peak_amp = max(rms[a], rms[b]) + 1e-9
            valley = seg.min()
            depths.append((peak_amp - valley) / peak_amp)
        result["labial_mod_depth"] = round(float(np.mean(depths)), 3) if depths else float("nan")
    else:
        result["labial_mod_depth"] = float("nan")

    return result


# ----------------------------------------------------------------------------
# 3. READING PASSAGE -> respiratory (phrase length), loudness, intelligibility
# ----------------------------------------------------------------------------
def analyze_reading(path, target_text=None, transcript=None):
    snd = load_sound(path)
    y, sr = librosa.load(path, sr=None, mono=True)

    # phrase length via silence segmentation (proxy for breath groups)
    intervals = librosa.effects.split(y, top_db=30)
    phrase_durs = [(e - s) / sr for s, e in intervals]
    result = {
        "mean_phrase_sec": round(float(np.mean(phrase_durs)), 2) if phrase_durs else float("nan"),
        "max_phrase_sec":  round(float(np.max(phrase_durs)), 2) if phrase_durs else float("nan"),
        "n_breath_groups": len(phrase_durs),
        "volume_std_db":   round(_voiced_intensity_std(snd), 2),
    }

    # intelligibility = word accuracy of transcript vs the KNOWN target lines.
    # transcript can be a clinician hand-transcription or any ASR output.
    if target_text and transcript:
        result["intelligibility_pct"] = round(_word_accuracy(target_text, transcript), 1)
    return result


def _word_accuracy(target, hyp):
    """1 - WER, expressed as %. Standard Levenshtein on word tokens."""
    t = target.lower().split()
    h = hyp.lower().split()
    d = [[0] * (len(h) + 1) for _ in range(len(t) + 1)]
    for i in range(len(t) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(t) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if t[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    wer = d[len(t)][len(h)] / max(len(t), 1)
    return max(0.0, (1 - wer)) * 100


# ----------------------------------------------------------------------------
# DOMAIN SCORING  (the "pattern" that turns raw metrics into 0-100 per domain)
# ----------------------------------------------------------------------------
def score_domains(vowel=None, ddk=None, reading=None, pataka=None):
    vowel = vowel or {}
    ddk = ddk or {}
    reading = reading or {}
    pataka = pataka or {}
    # merge for convenience; AMR /pa/ values take priority over SMR for shared keys
    raw = {**pataka, **ddk, **vowel, **reading}

    domains = {}

    # respiratory_support
    parts = [_score(raw.get("phonation_duration_sec"), **NORMS["phonation_duration_sec"])]
    if not np.isnan(reading.get("max_phrase_sec", float("nan"))):
        parts.append(_score(reading["max_phrase_sec"], good=8.0, poor=2.0))
    domains["respiratory_support"] = _avg(parts)

    # voice_stability
    domains["voice_stability"] = _avg([
        _score(raw.get("jitter_local_pct"),  **NORMS["jitter_local_pct"]),
        _score(raw.get("shimmer_local_pct"), **NORMS["shimmer_local_pct"]),
        _score(raw.get("hnr_db"),            **NORMS["hnr_db"]),
        _score(vowel.get("volume_std_db"),   **NORMS["volume_std_db_sustain"]),
    ])

    # labial_closure (lips, CN VII) -- prefer AMR /pa/, fall back to /pataka/ /pa/
    labial = ddk.get("labial_mod_depth")
    if labial is None or np.isnan(labial):
        labial = pataka.get("labial_mod_depth")
    domains["labial_closure"] = _score(labial, **NORMS["labial_mod_depth"])

    # lingual_closure (tongue, CN XII) -- only available from /pataka/ /ta/ /ka/
    domains["lingual_closure"] = _score(pataka.get("lingual_mod_depth"),
                                        **NORMS["lingual_mod_depth"])

    # ddk_regular -- combine AMR (/pa/) and SMR (/pataka/) when both present
    domains["ddk_regular"] = _avg([
        _score(ddk.get("ddk_cv_interval"),       **NORMS["ddk_cv_interval"]),
        _score(ddk.get("ddk_rate_syll_sec"),     **NORMS["ddk_rate_syll_sec"]),
        _score(pataka.get("smr_cv_interval"),    **NORMS["ddk_cv_interval"]),
        _score(pataka.get("smr_rate_syll_sec"),  **NORMS["smr_rate_syll_sec"]),
    ])

    # intelligibility
    if "intelligibility_pct" in raw:
        domains["intelligibility"] = _score(raw["intelligibility_pct"], **NORMS["intelligibility_pct"])
    else:
        domains["intelligibility"] = None

    return domains, raw


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None


# ----------------------------------------------------------------------------
# LONGITUDINAL TRACKER  (measure improvement across sessions)
# ----------------------------------------------------------------------------
class DysarthriaTracker:
    def __init__(self, store="patient_sessions.json"):
        self.store = store
        self.sessions = json.load(open(store)) if os.path.exists(store) else []

    def add_session(self, domains, raw, date=None, note=""):
        self.sessions.append({
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "domains": domains,
            "raw": _clean_nan(raw),
            "note": note,
        })
        json.dump(self.sessions, open(self.store, "w"), indent=2)

    def progress(self):
        """Delta of each domain between first and latest session."""
        if len(self.sessions) < 2:
            return {}
        first, last = self.sessions[0]["domains"], self.sessions[-1]["domains"]
        return {k: (None if first.get(k) is None or last.get(k) is None
                    else round(last[k] - first[k], 1)) for k in last}

    # ------------------------------------------------------------------
    # Longitudinal chart: six domains + the CN VII vs CN XII closure split
    # ------------------------------------------------------------------
    def plot_progress(self, out_path="progress.png", title="Dysarthria recovery"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator

        if not self.sessions:
            raise ValueError("No sessions logged yet.")

        labels = [s["date"] for s in self.sessions]
        x = list(range(len(self.sessions)))

        def series(domain):
            return [(s["domains"].get(domain) if s["domains"].get(domain) is not None
                     else float("nan")) for s in self.sessions]

        # six domains (order + colours)
        domain_style = {
            "respiratory_support": ("#3b6fb6", "Respiratory support"),
            "voice_stability":     ("#7a4fb5", "Voice stability"),
            "ddk_regular":         ("#c98a1b", "DDK regularity"),
            "intelligibility":     ("#4a4a4a", "Intelligibility"),
            "labial_closure":      ("#d1495b", "Labial closure (CN VII)"),
            "lingual_closure":     ("#119da4", "Lingual closure (CN XII)"),
        }

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 9), sharex=True,
            gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.18})

        # --- panel 1: all six domains -------------------------------------
        for dom, (color, lab) in domain_style.items():
            y = series(dom)
            lw = 2.6 if dom in ("labial_closure", "lingual_closure") else 1.8
            ax1.plot(x, y, marker="o", ms=5, lw=lw, color=color, label=lab)
        ax1.set_ylim(0, 105)
        ax1.yaxis.set_major_locator(MultipleLocator(20))
        ax1.set_ylabel("Domain score (0–100)")
        ax1.set_title(title + " — domain scores by session", fontweight="bold")
        ax1.grid(True, axis="y", ls=":", alpha=0.5)
        ax1.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)

        # --- panel 2: CN VII vs CN XII closure split, gap shaded ----------
        lab_y = np.array(series("labial_closure"), dtype=float)
        lin_y = np.array(series("lingual_closure"), dtype=float)
        ax2.plot(x, lab_y, marker="o", ms=6, lw=2.8, color="#d1495b",
                 label="Labial — CN VII (facial)")
        ax2.plot(x, lin_y, marker="s", ms=6, lw=2.8, color="#119da4",
                 label="Lingual — CN XII (hypoglossal)")
        # shade the divergence between the two articulators
        valid = ~(np.isnan(lab_y) | np.isnan(lin_y))
        ax2.fill_between(x, lab_y, lin_y, where=valid, alpha=0.18,
                         color="#888888", interpolate=True)
        # annotate the latest gap
        for i in range(len(x) - 1, -1, -1):
            if valid[i]:
                gap = lin_y[i] - lab_y[i]
                ax2.annotate(f"gap {gap:+.0f}",
                             (x[i], (lab_y[i] + lin_y[i]) / 2),
                             fontsize=9, ha="left", va="center",
                             xytext=(6, 0), textcoords="offset points")
                break
        ax2.set_ylim(0, 105)
        ax2.yaxis.set_major_locator(MultipleLocator(20))
        ax2.set_ylabel("Closure score (0–100)")
        ax2.set_title("Per-articulator closure: CN VII vs CN XII "
                      "(widening gap = differential recovery)", fontsize=10)
        ax2.grid(True, axis="y", ls=":", alpha=0.5)
        ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax2.set_xlabel("Session")

        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Small-multiples chart: one panel per raw metric, with the "good"
    # range (and a faint "poor" range) shaded straight from NORMS.
    # ------------------------------------------------------------------
    def plot_metrics(self, out_path="metrics.png", title="Raw metrics vs. normative ranges"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        if not self.sessions:
            raise ValueError("No sessions logged yet.")

        # (raw_key, NORMS_key, label).  Some raw keys reuse another metric's norm.
        METRICS = [
            ("phonation_duration_sec", "phonation_duration_sec", "Max phonation time (s)"),
            ("jitter_local_pct",       "jitter_local_pct",       "Jitter local (%)"),
            ("shimmer_local_pct",      "shimmer_local_pct",      "Shimmer local (%)"),
            ("hnr_db",                 "hnr_db",                 "HNR (dB)"),
            ("volume_std_db",          "volume_std_db_sustain",  "Intensity SD (dB)"),
            ("ddk_rate_syll_sec",      "ddk_rate_syll_sec",      "DDK rate /pa/ (syll/s)"),
            ("ddk_cv_interval",        "ddk_cv_interval",        "DDK CV /pa/"),
            ("smr_rate_syll_sec",      "smr_rate_syll_sec",      "SMR rate /pataka/ (syll/s)"),
            ("smr_cv_interval",        "ddk_cv_interval",        "SMR CV /pataka/"),
            ("labial_mod_depth",       "labial_mod_depth",       "Labial closure (CN VII)"),
            ("lingual_mod_depth",      "lingual_mod_depth",      "Lingual closure (CN XII)"),
            ("intelligibility_pct",    "intelligibility_pct",    "Intelligibility (%)"),
        ]

        labels = [s["date"] for s in self.sessions]
        x = list(range(len(self.sessions)))

        def series(raw_key):
            out = []
            for s in self.sessions:
                v = s.get("raw", {}).get(raw_key)
                out.append(float("nan") if v is None else float(v))
            return np.array(out, dtype=float)

        ncol = 3
        nrow = int(np.ceil(len(METRICS) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.0 * nrow))
        axes = np.array(axes).reshape(-1)

        GOOD, POOR, LINE = "#1f9d55", "#d1495b", "#23394d"

        for ax, (raw_key, norm_key, label) in zip(axes, METRICS):
            y = series(raw_key)
            good = NORMS[norm_key]["good"]
            poor = NORMS[norm_key]["poor"]
            higher_better = good >= poor

            finite = y[np.isfinite(y)]
            pool = list(finite) + [good, poor]
            lo, hi = min(pool), max(pool)
            pad = (hi - lo) * 0.12 or max(abs(hi), 1.0) * 0.12
            ylo, yhi = lo - pad, hi + pad
            ax.set_ylim(ylo, yhi)

            # shaded ranges
            if higher_better:
                ax.axhspan(good, yhi, color=GOOD, alpha=0.13)   # good zone
                ax.axhspan(ylo, poor, color=POOR, alpha=0.08)   # poor zone
            else:
                ax.axhspan(ylo, good, color=GOOD, alpha=0.13)
                ax.axhspan(poor, yhi, color=POOR, alpha=0.08)
            ax.axhline(good, color=GOOD, ls="--", lw=1.0, alpha=0.8)
            ax.axhline(poor, color=POOR, ls="--", lw=1.0, alpha=0.6)

            # patient trajectory
            ax.plot(x, y, "o-", color=LINE, lw=2.0, ms=5, zorder=5)

            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.grid(True, axis="y", ls=":", alpha=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            arrow = "↑ better" if higher_better else "↓ better"
            ax.text(0.02, 0.04, arrow, transform=ax.transAxes,
                    fontsize=7, color="#555", va="bottom")

        # hide any unused panels
        for ax in axes[len(METRICS):]:
            ax.set_visible(False)

        legend = [
            Patch(facecolor=GOOD, alpha=0.3, label="good range"),
            Patch(facecolor=POOR, alpha=0.2, label="poor range"),
            Line2D([0], [0], color=LINE, marker="o", label="patient (by session)"),
        ]
        fig.legend(handles=legend, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.005), fontsize=9, frameon=False)
        fig.suptitle(title, y=1.03, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return out_path


def _clean_nan(obj):
    """Recursively replace NaN floats with None so the JSON log stays valid."""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def full_session(vowel_path=None, ddk_path=None, reading_path=None,
                 target_text=None, transcript=None, ddk_syllable="pa",
                 pataka_path=None, reformat=True, sr=44100):
    """Run every available task and return (domains, raw_metrics).

    If reformat is True (default), each input is first normalised to mono 16-bit
    PCM WAV with ffmpeg so phone recordings (m4a/AAC/odd WAV) load reliably.
    """
    if reformat:
        vowel_path   = ensure_pcm_wav(vowel_path, sr)
        ddk_path     = ensure_pcm_wav(ddk_path, sr)
        reading_path = ensure_pcm_wav(reading_path, sr)
        pataka_path  = ensure_pcm_wav(pataka_path, sr)

    v = analyze_sustained_vowel(vowel_path) if vowel_path else None
    d = analyze_ddk(ddk_path, syllable=ddk_syllable) if ddk_path else None
    r = analyze_reading(reading_path, target_text, transcript) if reading_path else None
    pk = analyze_pataka(pataka_path) if pataka_path else None
    return score_domains(v, d, r, pk)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Flaccid dysarthria acoustic tracker")
    p.add_argument("--vowel"); p.add_argument("--ddk")
    p.add_argument("--pataka", help="/pataka/ SMR recording (per-articulator scoring)")
    p.add_argument("--reading"); p.add_argument("--target")
    p.add_argument("--transcript"); p.add_argument("--syllable", default="pa")
    p.add_argument("--log", default="patient_sessions.json")
    p.add_argument("--note", default="")
    p.add_argument("--plot", metavar="OUT.png",
                   help="render the longitudinal domain chart after logging")
    p.add_argument("--plot-metrics", metavar="OUT.png", dest="plot_metrics",
                   help="render the per-metric small-multiples chart with norm bands")
    p.add_argument("--reformat", action=argparse.BooleanOptionalAction, default=True,
                   help="auto-convert inputs to mono 16-bit PCM WAV via ffmpeg (default: on)")
    p.add_argument("--sr", type=int, default=44100, help="target sample rate for reformatting")
    a = p.parse_args()

    domains, raw = full_session(a.vowel, a.ddk, a.reading,
                                a.target, a.transcript, a.syllable, a.pataka,
                                reformat=a.reformat, sr=a.sr)
    print("RAW METRICS:");      print(json.dumps(_clean_nan(raw), indent=2))
    print("\nDOMAIN SCORES (0-100, higher = better):")
    print(json.dumps(domains, indent=2))

    t = DysarthriaTracker(a.log)
    t.add_session(domains, raw, note=a.note)
    if t.progress():
        print("\nPROGRESS vs first session:")
        print(json.dumps(t.progress(), indent=2))

    if a.plot:
        out = t.plot_progress(a.plot)
        print(f"\nChart saved to {out}")

    if a.plot_metrics:
        out = t.plot_metrics(a.plot_metrics)
        print(f"Per-metric chart saved to {out}")
