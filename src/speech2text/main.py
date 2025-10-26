import os
import queue
import sys
import tempfile
import threading

import dashscope
import keyboard as kb
import numpy as np
import sounddevice as sd
import wavio
from dashscope.audio.asr import Recognition
from loguru import logger
from pynput import keyboard

MODEL_NAME = "fun-asr-realtime"
SAMPLE_RATE = 16000
AUDIO_FORMAT = "wav"
SPEECH_FILE = "speech.wav"


class RealtimeASR:
    def __init__(self):
        self.audio_q = queue.Queue()
        self.recording_event = threading.Event()
        self.asr = Recognition(
            model=MODEL_NAME,
            callback=None,
            format=AUDIO_FORMAT,
            sample_rate=SAMPLE_RATE,
        )
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=self.audio_callback,
        )

    def audio_callback(self, indata, frames, time, status):
        if self.recording_event.is_set():
            self.audio_q.put(indata.copy())

    def start_stream(self):
        self.stream.start()

    def stop_stream(self):
        self.stream.stop()

    def recognize(self):
        frames = []
        while not self.audio_q.empty():
            frames.append(self.audio_q.get())
        if not frames:
            return

        audio_data = np.concatenate(frames, axis=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, SPEECH_FILE)
            wavio.write(wav_path, audio_data, SAMPLE_RATE, sampwidth=2)
            result = self.asr.call(wav_path)
            sentence = result.get_sentence()
            if sentence:
                text = sentence[0]["text"]
                logger.info(text)
                kb.write(text)

    def on_press(self, key):
        if key == keyboard.Key.ctrl_r:
            self.recording_event.set()

    def on_release(self, key):
        if key == keyboard.Key.ctrl_r:
            self.recording_event.clear()
            threading.Thread(target=self.recognize, daemon=True).start()


def init_dashscope_api_key():
    if "DASHSCOPE_API_KEY" in os.environ:
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    else:
        logger.error("DASHSCOPE_API_KEY environment variable is required.")
        sys.exit(1)


def main():
    init_dashscope_api_key()
    asr = RealtimeASR()
    asr.start_stream()

    with keyboard.Listener(
        on_press=asr.on_press, on_release=asr.on_release
    ) as listener:
        listener.join()


if __name__ == "__main__":
    main()
