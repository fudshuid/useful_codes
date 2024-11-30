#!/bin/bash

############ INFO ############
# Author : Gyu-Hwan Lee (Korea Institute of Science and Technology, Seoul National University)
# Contact: gh.lee@kist.re.kr
# Written : 2023-02-28
# Last edit : 2024-07-09 (Gyu-Hwan Lee)
# Description : Code for converting a recording of mouse ultrasonic vocalizations into human-audible one and listen it.
#               This code has one necessary input parameter and three optional input parameters:
#                   filepath: path to the audio file to be converted
#                   start, end: start and end time (in seconds) for the portion to convert (default: 10, 20)
#                   tempo: how fast you want to play (resulting in slow down effect) (default: 8x)
# Usage example:
# bash listen_usv.sh audio.wav 50 100
##############################

# command line inputs
filepath=$1
start=${2:-10}
end=${3:-20}
tempo=${4:-8}

echo "[INPUT] FILE: $filepath | START: $start (s) | END: $end (s)"

# cut the sound
echo "[PROGRESS] Cutting the desired audio segment using the input arguments..."
ffmpeg -loglevel error -y -i "$filepath" -ss "$start" -to "$end" -c copy sig.wav

# filter the sound
sox sig.wav filtered_high.wav sinc 40000-120000

# make sound louder
sox -v 10 filtered_high.wav filtered_high_louder.wav

# save slowed file
echo "[PROGRESS] Slow-winding the audio (pitch shifting)..."
ffmpeg -loglevel error -y -i filtered_high_louder.wav -af "asetrate=10000" resampled.wav

# shorten the play time (without pitch shifting)
echo "[PROGRESS] Fast-winding the audio..."
ffmpeg -loglevel error -y -i resampled.wav -af "atempo=$tempo" output.wav

# delete intermediate files
rm sig.wav filtered_high.wav filtered_high_louder.wav resampled.wav

# play the resulting file
echo "[PROGRESS] Resulting audio will be played along with a spectrogram window"
ffplay -loglevel error -showmode 2 -left 0 -top 0 -window_title "Sound Display" -x 1200 -alwaysontop -autoexit output.wav

# when playing is done, remove the result file
rm output.wav
echo "[FINISHED]"
