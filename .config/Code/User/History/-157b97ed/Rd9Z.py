import pyaudio
import simpleaudio as sa
import numpy as np
import time

# def play_beep(frequency, duration=0.1):
#     sample_rate = 44100  # samples per second
#     t = np.linspace(0, duration, int(sample_rate * duration), False)
#     tone = 0.5 * np.sin(frequency * t * 2 * np.pi)
#     audio = tone.astype(np.float32).tobytes()
#     play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
#     play_obj.wait_done()
    
import numpy as np
import pyaudio
import threading
from concurrent.futures import ThreadPoolExecutor

class AudioPlayer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=True)

    def play_beep(self, frequency, duration=0.1, volume=0.5):
        def play_thread():
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            sine_wave = 0.5 * volume * (np.sin(2 * np.pi * frequency * t)).astype(np.float32)
            self.stream.write(sine_wave.tobytes())

        self.executor.submit(play_thread)


# Test play_beep
if __name__ == "__main__":
    # pygame.mixer.init()
    # frequency = 440  # Our played note will be 440 Hz
    # duration = 3.0  # In seconds, may be float
    # play_beep(frequency, duration)

    # Example usage
    audio_player = AudioPlayer()
    audio_player.play_beep(440, duration=0.5)  # Play a 440 Hz beep for 0.1 seconds (default duration)
    # audio_player.play_beep(440, duration=0.3)  # Play a 440 Hz beep for 0.1 seconds (default duration)
    # Play additional beeps as needed
    audio_player.executor.shutdown(wait=True)
    audio_player.stream.stop_stream()
    audio_player.stream.close()
    audio_player.p.terminate()