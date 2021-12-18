import argparse
import queue
import sys
import math
import struct
import wave
import time
import os
import base64
import requests
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


Threshold = 10

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
# FORMAT = sd
CHANNELS = 1
RATE = 16000
swidth = 2

TIMEOUT_LENGTH = 5

# todo add to spotify playlist via api?

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


class Listener:
    def __init__(self):
        self.parser = argparse.ArgumentParser(add_help=False)
        self.parser.add_argument(
            '-l', '--list-devices', action='store_true',
            help='show list of audio devices and exit')
        args, remaining = self.parser.parse_known_args()
        if args.list_devices:
            print(sd.query_devices())
            self.parser.exit(0)
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[self.parser])
        parser.add_argument(
            'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
            help='input channels to plot (default: the first)')
        parser.add_argument(
            '-d', '--device', type=int_or_str,
            help='input device (numeric ID or substring)')
        parser.add_argument(
            '-w', '--window', type=float, default=200, metavar='DURATION',
            help='visible time slot (default: %(default)s ms)')
        parser.add_argument(
            '-i', '--interval', type=float, default=30,
            help='minimum time between plot updates (default: %(default)s ms)')
        parser.add_argument(
            '-b', '--blocksize', type=int, help='block size (in samples)')
        parser.add_argument(
            '-r', '--samplerate', type=float, help='sampling rate of audio device')
        parser.add_argument(
            '-n', '--downsample', type=int, default=10, metavar='N',
            help='display every Nth sample (default: %(default)s)')
        args = parser.parse_args(remaining)
        if any(c < 1 for c in args.channels):
            parser.error('argument CHANNEL: must be >= 1')
        self.mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
        self.q = queue.Queue()
        self.downsample = args.downsample
        self.channels = args.channels
        self.samplerate = args.samplerate
        self.device = args.device
        self.window = args.window
        self.interval = args.interval
        self.blocksize = args.blocksize
        self.plotdata = np.zeros((100, len(self.channels)))
        self.lines = []
        self.threshold = 75
        self.recordingFrameTotal = 100
        self.recordingFrameCount = 0
        self.record = False
        self.recording_frames = []
        self.songs = []


    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        volume_norm = np.linalg.norm(indata) * 10
        if not self.record and volume_norm > self.threshold:
            self.record = True
            print("Over Threshold")
            print("|" * int(volume_norm))

        if self.record:
            self.update_recording(indata)
        self.q.put(indata[::self.downsample, self.mapping])

    def update_recording(self, frame):
        if self.recordingFrameCount < self.recordingFrameTotal:
            self.recording_frames.append(frame)
            self.recordingFrameCount += 1
        else:
            self.recordingFrameCount = 0
            self.record = False
            self.encode_send_frames_and_update_songs()
            self.recording_frames = []

    def encode_send_frames_and_update_songs(self):
        encoded_frames = self.recording_frames
        # 44100Hz, 1 channel (Mono), signed 16 bit PCM little endian
        # base 64 encode as string
        # todo encode correctly?
        # todo base64 encode to string correctly
        encoded_frames = base64.encodebytes(encoded_frames)
        song = self.get_song_from_shazam(encoded_frames)
        if song:
            print(f"found song: {song}")
            self.songs.append(song)

    def update_plot(self, frame):
        """This is called by matplotlib for each plot update.

        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.

        """
        while True:
            try:
                data = self.q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data
        for column, line in enumerate(self.lines):
            line.set_ydata(self.plotdata[:, column])
        return self.lines

    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def listen(self):
        try:
            if self.samplerate is None:
                device_info = sd.query_devices(self.device, 'input')
                self.samplerate = device_info['default_samplerate']

            length = int(self.window * self.samplerate / (1000 * self.downsample))
            self.plotdata = np.zeros((length, len(self.channels)))

            fig, ax = plt.subplots()
            self.lines = ax.plot(self.plotdata)
            if len(self.channels) > 1:
                ax.legend(['channel {}'.format(c) for c in self.channels],
                          loc='lower left', ncol=len(self.channels))
            ax.axis((0, len(self.plotdata), -1, 1))
            ax.set_yticks([0])
            ax.yaxis.grid(True)
            ax.tick_params(bottom=False, top=False, labelbottom=False,
                           right=False, left=False, labelleft=False)
            fig.tight_layout(pad=0)
            input(f"device: {self.device}, channels: {self.channels}, samplerate: {self.samplerate}. continue?")
            # todo need to use raw input stream?
            # RawInputStream
            stream = sd.InputStream(
                device=self.device, channels=max(self.channels),
                samplerate=self.samplerate, callback=self.audio_callback)
            ani = FuncAnimation(fig, self.update_plot, interval=self.interval, blit=True)
            with stream:
                plt.show()
        except Exception as e:
            self.parser.exit(type(e).__name__ + ': ' + str(e))

    @staticmethod
    def get_song_from_shazam(audio):
        url = "https://shazam.p.rapidapi.com/songs/v2/detect"

        querystring = {"timezone": "America/Chicago", "locale": "en-US",
                       }

        headers = {
            'x-rapidapi-host': "shazam.p.rapidapi.com",
            'x-rapidapi-key': "f516d1081fmshf0d4b17687c4683p1c4285jsnfc42f35f09a7",
            'Content-Type': 'text/plain'
        }
        # todo uncomment to actually hit the api
        # response = requests.request("POST", url, data=audio, headers=headers, params=querystring)
        # print(response.status_code)
        # print(response.text)
        # if response.status_code < 300:
        #     return response.json()
        # print("Song not found :(")
