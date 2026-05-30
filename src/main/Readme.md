


# inputs converter

## installation 

(Install ffmpeg if needed: brew install ffmpeg on Mac, sudo apt install ffmpeg on Linux, or winget/choco on Windows.)

sudo apt install ffmpeg

## usage

ffmpeg -y -i recording.m4a -ac 1 -ar 44100 -c:a pcm_s16le recording.wav

-ac 1 forces mono, -ar 44100 sets the sample rate, 
-c:a pcm_s16le writes standard 16-bit WAV. 