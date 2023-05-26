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