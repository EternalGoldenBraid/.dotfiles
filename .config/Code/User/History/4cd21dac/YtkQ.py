# %% [markdown]
# # Whisper's transcription plus Pyannote's Diarization 
# 
# Andrej Karpathy [suggested](https://twitter.com/karpathy/status/1574476200801538048?s=20&t=s5IMMXOYjBI6-91dib6w8g) training a classifier on top of  OpenAI [Whisper](https://openai.com/blog/whisper/) model features to identify the speaker, so we can visualize the speaker in the transcript. But, as [pointed out](https://twitter.com/tarantulae/status/1574493613362388992?s=20&t=s5IMMXOYjBI6-91dib6w8g) by Christian Perone, it seems that features from whisper wouldn't be that great for speaker recognition as its main objective is basically to ignore speaker differences.
# 
# In the following, I use [**`pyannote-audio`**](https://github.com/pyannote/pyannote-audio), a speaker diarization toolkit by Hervé Bredin, to identify the speakers, and then match it with the transcriptions of Whispr. I do it on the first 30 minutes of  Lex's 2nd [interview](https://youtu.be/SGzMElJ11Cc) with Yann LeCun. Check the result [**here**](https://majdoddin.github.io/lexicap.html). 
# 
# It is tricky to match the transcriptions to diarization segemtns, specially when the speaker changes. To resolve it, Sarah Kaiser [suggested](https://github.com/openai/whisper/discussions/264#discussioncomment-3825375) runnnig the pyannote.audio first and  then just running whisper on the split-by-speaker chunks. 
# For sake of performance (and transcription quality?), we attach the audio segements into a single audio file with a silent spacer as a seperator, and run whisper on it. Enjoy it!

# %% [markdown]
# # Preparing the audio file

# %%
from pydub import AudioSegment

# %%

t1 = 0 * 1000 #Works in milliseconds
t2 = 20 * 60 * 1000

newAudio = AudioSegment.from_wav("download.wav")
a = newAudio[t1:t2]
a.export("lecun1.wav", format="wav") 


# %% [markdown]
# `pyannote.audio` seems to miss the first 0.5 seconds of the audio, and, therefore, we prepend a spcacer.

# %%
audio = AudioSegment.from_wav("lecun1.wav")
spacermilli = 2000
spacer = AudioSegment.silent(duration=spacermilli)
audio = spacer.append(audio, crossfade=0)

audio.export('audio.wav', format='wav')

# %%
# audio = AudioSegment.from_ogg("data/audio.ogg")
# audio.export('audio.wav', format='wav')
# audio

# %% [markdown]
# # Pyannote's Diarization

# %% [markdown]
# [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) is an open-source toolkit written in Python for **speaker diarization**. 
# 
# Based on [`PyTorch`](https://pytorch.org) machine learning framework, it provides a set of trainable end-to-end neural building blocks that can be combined and jointly optimized to build speaker diarization pipelines. 
# 
# `pyannote.audio` also comes with pretrained [models](https://huggingface.co/models?other=pyannote-audio-model) and [pipelines](https://huggingface.co/models?other=pyannote-audio-pipeline) covering a wide range of domains for voice activity detection, speaker segmentation, overlapped speech detection, speaker embedding reaching state-of-the-art performance for most of them. 

# %% [markdown]
# Installing Pyannote and running it on the video to generate the diarizations.

# %%
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization')

# %%
DEMO_FILE = {'uri': 'blabal', 'audio': 'audio.wav'}
dz = pipeline(DEMO_FILE,
              use_auth_token='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')  

with open("diarization.txt", "w") as text_file:
    text_file.write(str(dz))

# %%
print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")

# %%
def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

# %%
import re
dz = open('diarization.txt').read().splitlines()
dzList = []
for l in dz:
  start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
  start = millisec(start) - spacermilli
  end = millisec(end)  - spacermilli
  lex = not re.findall('SPEAKER_01', string=l)
  dzList.append([start, end, lex])

print(*dzList[:10], sep='\n')

# %% [markdown]
# # Preparing audio file from the diarization

# %% [markdown]
# Attaching audio segements according to the diarization, with a spacer as the delimiter.

# %%
from pydub import AudioSegment
import re 

sounds = spacer
segments = []

dz = open('diarization.txt').read().splitlines()
for l in dz:
  start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
  start = int(millisec(start)) #milliseconds
  end = int(millisec(end))  #milliseconds
  
  segments.append(len(sounds))
  sounds = sounds.append(audio[start:end], crossfade=0)
  sounds = sounds.append(spacer, crossfade=0)

sounds.export("dz.wav", format="wav") #Exports to a wav file in the current path.

# %%
segments[:8]

# %% [markdown]
# Freeing up some memory

# %%
del   sounds, DEMO_FILE, pipeline, spacer,  audio, dz, a, newAudio

# %% [markdown]
# # Whisper's Transcriptions

# %% [markdown]
# Installing Open AI whisper.
# 
# **Important:** There is a version conflict with pyannote.audio resulting in an error (see this RP). Our workaround is to first run Pyannote and then whisper. You can safely ignore the error.
# 
# %% [markdown]
# Running Open AI whisper on the prepared audio file. [link text](https://) It writes the transcription into a file.

# %%
os.cmd('whisper dz.wav --language en --model large')

# %% [markdown]
# Reading the transcription file.

# %%
!pip install -U webvtt-py

# %%
import webvtt

captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read('dz.wav.vtt')]
print(*captions[:8], sep='\n')

# %% [markdown]
# # Matching the Transcriptions and the Diarizations

# %% [markdown]
# Matching each trainscrition line to some diarizations, and generating the HTML file. To get the correct timing, we should take care of the parts in original audio that were in no diarization segment.

# %%
preS = '<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <meta http-equiv="X-UA-Compatible" content="ie=edge">\n    <title>Lexicap</title>\n    <style>\n        body {\n            font-family: sans-serif;\n            font-size: 18px;\n            color: #111;\n            padding: 0 0 1em 0;\n        }\n        .l {\n          color: #050;\n        }\n        .s {\n            display: inline-block;\n        }\n        .e {\n            display: inline-block;\n        }\n        .t {\n            display: inline-block;\n        }\n        #player {\n\t\tposition: sticky;\n\t\ttop: 20px;\n\t\tfloat: right;\n\t}\n    </style>\n  </head>\n  <body>\n    <h2>Yann LeCun: Dark Matter of Intelligence and Self-Supervised Learning | Lex Fridman Podcast #258</h2>\n  <div  id="player"></div>\n    <script>\n      var tag = document.createElement(\'script\');\n      tag.src = "https://www.youtube.com/iframe_api";\n      var firstScriptTag = document.getElementsByTagName(\'script\')[0];\n      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);\n      var player;\n      function onYouTubeIframeAPIReady() {\n        player = new YT.Player(\'player\', {\n          height: \'210\',\n          width: \'340\',\n          videoId: \'SGzMElJ11Cc\',\n        });\n      }\n      function setCurrentTime(timepoint) {\n        player.seekTo(timepoint);\n   player.playVideo();\n   }\n    </script><br>\n'
postS = '\t</body>\n</html>'

# %%
from datetime import timedelta

html = list(preS)

for i in range(len(segments)):
  idx = 0
  for idx in range(len(captions)):
    if captions[idx][0] >= (segments[i] - spacermilli):
      break;
  
  while (idx < (len(captions))) and ((i == len(segments) - 1) or (captions[idx][1] < segments[i+1])):
    c = captions[idx]  
    
    start = dzList[i][0] + (c[0] -segments[i])

    if start < 0: 
      start = 0
    idx += 1

    start = start / 1000.0
    startStr = '{0:02d}:{1:02d}:{2:02.2f}'.format((int)(start // 3600), 
                                            (int)(start % 3600 // 60), 
                                            start % 60)
    
    html.append('\t\t\t<div class="c">\n')
    html.append(f'\t\t\t\t<a class="l" href="#{startStr}" id="{startStr}">link</a> |\n')
    html.append(f'\t\t\t\t<div class="s"><a href="javascript:void(0);" onclick=setCurrentTime({int(start)})>{startStr}</a></div>\n')
    html.append(f'\t\t\t\t<div class="t">{"[Lex]" if dzList[i][2] else "[Yann]"} {c[2]}</div>\n')
    html.append('\t\t\t</div>\n\n')

html.append(postS)
s = "".join(html)

with open("lexicap.html", "w") as text_file:
    text_file.write(s)
print(s)


