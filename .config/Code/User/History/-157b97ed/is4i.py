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
    
import pygame


def play_beep(frequency, duration=0.1):
    sample_rate = 44100
    samples = np.array([np.sin(2.0 * np.pi * frequency * x / sample_rate) for x in range(int(sample_rate * duration))]).astype(np.int16)
    stereo_samples = np.array([samples, samples]).T.copy(order='C') # Create a stereo array by duplicating the samples
    tone = pygame.sndarray.make_sound(stereo_samples)
    tone.play()
    
class AudioPlayer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=2,
                                  rate=self.sample_rate,
                                  output=True)

    def play_beep(self, frequency, duration=0.1, A=0.1):
        samples = A*np.array([np.sin(2.0 * np.pi * frequency * x / self.sample_rate) for x in range(int(self.sample_rate * duration))]).astype(np.float32)
        stereo_samples = np.array([samples, samples]).T
        self.stream.write(stereo_samples.tobytes())

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    

# Test play_beep
if __name__ == "__main__":
    # pygame.mixer.init()
    # frequency = 440  # Our played note will be 440 Hz
    # duration = 3.0  # In seconds, may be float
    # play_beep(frequency, duration)

    # Example usage
    audio_player = AudioPlayer()
    audio_player.play_beep(440, duration=0.5)  # Play a 440 Hz beep for 0.1 seconds (default duration)
    # Play additional beeps as needed
    audio_player.close()  # Close the audio stream when done