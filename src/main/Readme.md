


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

$ python ./src/main/python/dysarthria_analysis.py   --vowel ./src/main/resources/recordings/Aaaaa-AUDIO-2026-05-30-11-46-42.wav --ddk ./src/main/resources/recordings/papapa-AUDIO-2026-05-30-13-02-48.wav --pataka ./src/main/resources/recordings/pataka-AUDIO-2026-05-30-11-47-35.wav --reading ./src/main/resources/recordings/Reading-AUDIO-2026-05-30-11-44-42.wav --note "first try from Jose" --plot "OUT.png" --plot-metrics "OUT.png" \
--target "Mi papá compró pan y pasteles para la fiesta.Mamá puso la mesa con platos y tazas.Pablo y Pepe tomaron café con tostadas.Todos quedaron muy contentos después de comer." \
[--transcript "Mi papá compró pan y pasteles para la fiesta.Mamá puso la mesa con platos y tazas.Pablo y Pepe tomaron café con tostadas.Todos quedaron 
muy contentos después de comer."]

# exploring m4a inputs

$ python ./src/main/python/dysarthria_analysis.py   --vowel ./src/main/resources/recordings/Aaaaa-AUDIO-2026-05-30-11-46-42.m4a --ddk ./src/main/resources/recordings/papapa-AUDIO-2026-05-30-13-02-48.m4a --pataka ./src/main/resources/recordings/pataka-AUDIO-2026-05-30-11-47-35.m4a --reading ./src/main/resources/recordings/Reading-AUDIO-2026-05-30-11-44-42.m4a --note "first try from Jose" --plot "OUT.png" --plot-metrics "OUT.png" \
--target "Mi papá compró pan y pasteles para la fiesta.Mamá puso la mesa con platos y tazas.Pablo y Pepe tomaron café con tostadas.Todos quedaron muy contentos después de comer." \
--transcript "Mi papá compró pan y pasteles para la fiesta.Mamá puso la mesa con platos y tazas.Pablo y Pepe tomaron café con tostadas.Todos quedaron muy contentos después de comer."