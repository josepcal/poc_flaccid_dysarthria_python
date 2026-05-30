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
import statistics
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
    "labial_mod_depth":       {"good": 0.85, "poor": 0.35},  # higher better (full closure)
    "intelligibility_pct":    {"good": 95.0, "poor": 50.0},
}


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
def load_sound(path, target_sr=44100):
    snd = parselmouth.Sound(path)
    return snd


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
# 2. DDK  ->  ddk_regular + labial_closure
# ----------------------------------------------------------------------------
def analyze_ddk(path, syllable="pa"):
    """Detects syllable onsets in a /pa/ (or /pataka/) repetition task."""
    y, sr = librosa.load(path, sr=None, mono=True)

    # amplitude envelope -> onset peaks
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    times = librosa.times_like(env, sr=sr, hop_length=256)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5,
                                   post_avg=5, delta=np.median(env) * 0.6, wait=4)
    onsets = times[peaks]

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
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
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
def score_domains(vowel=None, ddk=None, reading=None):
    vowel = vowel or {}
    ddk = ddk or {}
    reading = reading or {}
    raw = {**vowel, **ddk, **reading}

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

    # labial_closure
    domains["labial_closure"] = _score(raw.get("labial_mod_depth"), **NORMS["labial_mod_depth"])

    # ddk_regular
    domains["ddk_regular"] = _avg([
        _score(raw.get("ddk_cv_interval"),   **NORMS["ddk_cv_interval"]),
        _score(raw.get("ddk_rate_syll_sec"), **NORMS["ddk_rate_syll_sec"]),
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
            "raw": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in raw.items()},
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


def full_session(vowel_path=None, ddk_path=None, reading_path=None,
                 target_text=None, transcript=None, ddk_syllable="pa"):
    """Run every available task and return (domains, raw_metrics)."""
    v = analyze_sustained_vowel(vowel_path) if vowel_path else None
    d = analyze_ddk(ddk_path, syllable=ddk_syllable) if ddk_path else None
    r = analyze_reading(reading_path, target_text, transcript) if reading_path else None
    return score_domains(v, d, r)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Flaccid dysarthria acoustic tracker")
    p.add_argument("--vowel"); p.add_argument("--ddk")
    p.add_argument("--reading"); p.add_argument("--target")
    p.add_argument("--transcript"); p.add_argument("--syllable", default="pa")
    p.add_argument("--log", default="patient_sessions.json")
    p.add_argument("--note", default="")
    a = p.parse_args()

    domains, raw = full_session(a.vowel, a.ddk, a.reading,
                                a.target, a.transcript, a.syllable)
    print("RAW METRICS:");      print(json.dumps(raw, indent=2))
    print("\nDOMAIN SCORES (0-100, higher = better):")
    print(json.dumps(domains, indent=2))

    t = DysarthriaTracker(a.log)
    t.add_session(domains, raw, note=a.note)
    if t.progress():
        print("\nPROGRESS vs first session:")
        print(json.dumps(t.progress(), indent=2))