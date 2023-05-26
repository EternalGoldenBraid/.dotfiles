import simpleaudio as sa
import numpy as np
import time

def play_beep(frequency, duration=0.1):
    sample_rate = 44100  # samples per second
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(frequency * t * 2 * np.pi)
    audio = tone.astype(np.float32).tobytes()
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    play_obj.wait_done()