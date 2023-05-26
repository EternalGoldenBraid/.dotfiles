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
    os.system(f'yt-dlp -xv --ffmpeg-location ffmpeg-master-latest-linux64-gpl/bin --audio-format wav  -o {output} -- {url}')

def cut_audio(input_file='download.wav', output_file='audio.wav', start=0, end=10):
    """
    Let's cut the first 20 minutes of the audio

    Args:
        input_file (str, optional): Input file. Defaults to 'download.wav'.
        output_file (str, optional): Output file. Defaults to 'audio.wav'.
        start (int, optional): Start time in seconds. Defaults to 0.
        end (int, optional): End time in minutes. Defaults to 10.
    """
    t1 = start * 1000 #Works in milliseconds
    t2 = end * 60 * 1000
    
    audio = AudioSegment.from_wav(input_file)
    a = audio[t1:t2]
    # a.export("lecun1.wav", format="wav") 

    # pyannote.audio seems to miss the first 0.5 seconds of the audio,
    # and, therefore, we prepend a spcacer.
    # audio = AudioSegment.from_wav("lecun1.wav")
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = spacer.append(audio, crossfade=0)
    
    audio.export(output_file, format='wav')
    
    return audio, spacer, spacermilli


if __name__ == '__main__':
    download_youtube(output='test.wav', url='https://www.youtube.com/watch?v=DNG9VJ5j1D4')
    audio, spacer, spacermilli = cut_audio()
    os.system('vlc test.wav')
    # pipeline = Pipeline()
    # pipeline = Pipeline()