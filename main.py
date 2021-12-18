import requests
import sounddevice as sd
from listener import Listener
import numpy as np



def capture_song():
    duration = 5  # seconds
    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        volume_norm = np.linalg.norm(indata) * 10
        print("|" * int(volume_norm))
        outdata[:] = indata
    while True:
        with sd.Stream(channels=1, callback=callback):
            sd.sleep(int(duration * 1000))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # detect_song()
    # capture_song()
    l = Listener()
    l.listen()
