


# inputs converter

## installation 

(Install ffmpeg if needed: brew install ffmpeg on Mac, sudo apt install ffmpeg on Linux, or winget/choco on Windows.)

sudo apt install ffmpeg

## usage

ffmpeg -y -i recording.m4a -ac 1 -ar 44100 -c:a pcm_s16le recording.wav

-ac 1 forces mono, -ar 44100 sets the sample rate, 
-c:a pcm_s16le writes standard 16-bit WAV. 


# installing python libs

cd ~/codebase/PoCs/Poc_Flaccid_Dysarthria
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy librosa soundfile praat-parselmouth matplotlib


# exploring inputs

$ python ./src/main/python/dysarthria_analysis.py   --vowel ./src/main/resources/recordings/Aaaaa-AUDIO-2026-05-30-11-46-42.wav --pataka ./src/main/resources/recordings/pataka-AUDIO-2026-05-30-11-47-35.wav --reading ./src/main/resources/recordings/Reading-AUDIO-2026-05-30-11-44-42.wav --note "first try from Jose" --plot "OUT.png" \
  [--target "the early bird catches the worm in the morning"] \
  [--transcript "the early bird catches the worm in the morning"]