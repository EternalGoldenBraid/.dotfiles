from typing import List, Tuple, Dict, Any
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
from threading import Lock
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
        
# class AudioPlayerParallel:
#     def __init__(self):
#         self.p = pyaudio.PyAudio()
#         self.sample_rate = 44100
#         self.lock = Lock()

#     def beep(self, frequency, duration, volume=0.05):
#         def play_beep():
#             with self.lock:
#                 # Generate the beep samples
#                 num_samples = int(self.sample_rate * duration)
#                 samples = (np.sin(2 * np.pi * np.arange(num_samples) * frequency / self.sample_rate)).astype(np.float32)

#                 # Open a streaming stream
#                 stream = self.p.open(format=pyaudio.paFloat32,
#                                      channels=1,
#                                      rate=self.sample_rate,
#                                      output=True)

#                 # Play the beep
#                 stream.write(volume * samples)

#                 # Close the stream
#                 stream.stop_stream()
#                 stream.close()

#         # Start a new thread to play the beep
#         beep_thread = threading.Thread(target=play_beep)
#         beep_thread.start()

#     def close(self):
#         self.p.terminate()
        
# class AudioPlayerParallel:
#     def __init__(self):
#         self.p = pyaudio.PyAudio()
#         info = self.p.get_host_api_info_by_index(0)
#         numdevices = info.get('deviceCount')
#         self.sample_rate = 44100
#         self.playing = False

#     def beep(self, frequency, duration, volume=0.2):
#         if self.playing:
#             return
#         def play_beep():
#             self.playing = True
#             # Generate the beep samples
#             num_samples = int(self.sample_rate * duration)
#             samples = (np.sin(2 * np.pi * np.arange(num_samples) * frequency / self.sample_rate)).astype(np.float32)

#             # Open a streaming stream
#             stream = self.p.open(format=pyaudio.paFloat32,
#                                  channels=1,
#                                  rate=self.sample_rate,
#                                  output=True)

#             # Play the beep
#             stream.write(volume * samples)

#             # Close the stream
#             stream.stop_stream()
#             stream.close()
#             self.playing = False

#         # Start a new thread to play the beep
#         beep_thread = threading.Thread(target=play_beep)
#         beep_thread.start()

    def close(self):
        self.p.terminate()
        
import numpy as np
import pygame

# class AudioPlayerParallel:
#     def __init__(self):
#         pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)
#         self.sample_rate = 44100

#     def beep(self, frequency, duration, volume=0.2):
#         num_samples = int(self.sample_rate * duration)
#         samples = (np.sin(2 * np.pi * np.arange(num_samples) * frequency / self.sample_rate)).astype(np.float32)
#         samples = (volume * samples * 32767).astype(np.int16)  # Convert to 16-bit signed integer
#         sound = pygame.sndarray.make_sound(samples)
#         sound.play()
        
#     def play_chord(self, frequencies, duration, volumes):
#         num_samples = int(self.sample_rate * duration)
#         chord_samples = np.zeros(num_samples, dtype=np.float32)

#         for frequency, volume in zip(frequencies, volumes):
#             samples = (np.sin(2 * np.pi * np.arange(num_samples) * frequency / self.sample_rate)).astype(np.float32)
#             samples = (volume * samples * 32767).astype(np.int16)  # Convert to 16-bit signed integer
#             chord_samples += samples

#         chord_samples = np.clip(chord_samples, -32767, 32767).astype(np.int16)  # Clip to avoid overflow
#         sound = pygame.sndarray.make_sound(chord_samples)
#         sound.play()


#     def close(self):
#         pygame.mixer.quit()
        
class AudioPlayerParallel:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)
        self.sample_rate = 44100

    def beep(self, frequency: float, duration: float, volume: float = 0.2):
        num_samples = int(self.sample_rate * duration)
        samples = (np.sin(2 * np.pi * np.arange(num_samples) * frequency / self.sample_rate)).astype(np.float32)
        samples = (volume * samples * 32767).astype(np.int16)  # Convert to 16-bit signed integer
        sound = pygame.sndarray.make_sound(samples)
        sound.play()

    def play_chord(self, frequencies: List[float], durations: List[float], volumes: List[float]):
        max_duration = max(durations)
        num_samples = int(self.sample_rate * max_duration)
        chord_samples = np.zeros(num_samples, dtype=np.float32)

        for frequency, duration, volume in zip(frequencies, durations, volumes):
            num_samples_frequency = int(self.sample_rate * duration)
            samples = (np.sin(2 * np.pi * np.arange(num_samples_frequency) * frequency / self.sample_rate)).astype(np.float32)
            samples = (volume * samples * 32767).astype(np.int16)  # Convert to 16-bit signed integer
            chord_samples[:num_samples_frequency] += samples

        chord_samples = np.clip(chord_samples, -32767, 32767).astype(np.int16)  # Clip to avoid overflow
        sound = pygame.sndarray.make_sound(chord_samples)
        sound.play()

    def close(self):
        pygame.mixer.quit()



# Test play_beep
if __name__ == "__main__":
    # pygame.mixer.init()
    # frequency = 440  # Our played note will be 440 Hz
    # duration = 3.0  # In seconds, may be float
    # play_beep(frequency, duration)

    # Example usage
    # audio_player = AudioPlayer()
    # audio_player.play_beep(440, duration=0.5)  # Play a 440 Hz beep for 0.1 seconds (default duration)
    # audio_player.executor.shutdown(wait=True)
    # audio_player.stream.stop_stream()
    # audio_player.stream.close()
    # audio_player.p.terminate()
    
    import time
    audio_player = AudioPlayerParallel()
    
    # Play beeps for multiple ribosomes simultaneously
    audio_player.beep(440, 0.5)  # Ribosome 1
    audio_player.beep(660, 0.5)  # Ribosome 2
    audio_player.beep(880, 0.5)  # Ribosome 3
    
    time.sleep(1)  # Wait for the beeps to finish
    
    audio_player.close()