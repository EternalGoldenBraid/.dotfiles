from datetime import timedelta
from pydub import AudioSegment
from pyannote.audio import Pipeline
import webvtt
import sys
import os
import re
from pydub import AudioSegment
from tqdm import tqdm

def download_youtube(url, output='test'):
    # os.system(f'yt-dlp -xv --ffmpeg-location ffmpeg-master-latest-linux64-gpl/bin --audio-format wav  -o download.wav -- https://youtu.be/SGzMElJ11Cc')
    os.system(f'yt-dlp -xv --ffmpeg-location ffmpeg-master-latest-linux64-gpl/bin --audio-format wav  -o {output}.wav -- {url}')

def cut_audio():
    "Let's cut the first 20 minutes of the audio"
    minutes = 5
    t1 = 0 * 1000 #Works in milliseconds
    t2 = minutes * 60 * 1000
    
    newAudio = AudioSegment.from_wav("download.wav")
    a = newAudio[t1:t2]
    a.export("lecun1.wav", format="wav") 

    # pyannote.audio seems to miss the first 0.5 seconds of the audio,
    # and, therefore, we prepend a spcacer.
    audio = AudioSegment.from_wav("lecun1.wav")
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = spacer.append(audio, crossfade=0)
    
    audio.export('audio.wav', format='wav')
    
    return audio, spacer, spacermilli