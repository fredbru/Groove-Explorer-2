from pydub import AudioSegment
import simpleaudio
import time

path = 'static/Part 1 MP3/'

class audio_player(object):
    def __init__(self):
        pass

    def play_audio(self, file_name):
        sound = AudioSegment.from_mp3(file_name)
        time.sleep(0.05)
        self.playback = simpleaudio.play_buffer(
            sound.raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate
        )

    def stop_audio(self):
        try:
            self.playback.stop()
            time.sleep(0.1)
        except:
            pass

