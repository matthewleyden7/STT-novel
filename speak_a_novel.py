import onnx
import torch
import onnxruntime
from omegaconf import OmegaConf
import time
import math
import audioop
from collections import deque
import pyaudio
import wave
import os
import sys
import numpy as np
from subprocess import PIPE, Popen
import sounddevice as sd
import random
from fastpunct import FastPunct
fastpunct = FastPunct()
# Import sometimes fails first time around because of a Cython issue.





# Options
CHUNK = 128 # The size of each audio chunk coming from the input device.
FORMAT = pyaudio.paInt16 # Should not be changed, as this format is best for speech recognition.
RATE = 16000 # Speech recognition only works well with this rate.  Don't change unless your microphone demands it.
RECORD_SECONDS = 5 # Number of seconds to record, can be changed.
WAVE_OUTPUT_FILENAME = "output.wav" # Where to save the recording from the microphone.
CHANNELS = 2

class VolumeContainer:
    def __init__(self):
        self.container = []
        self.max_min_diff = 0
        self.avg = 0
        self.all_avgs = []
        self.timer = time.time()
        self.silence_time = 1.5
        self.silence_threshold = 3000
        self.sensitivity = 20.0

    def check_noise_lvl(self, data, outdata, frames, time, status):
        volume_norm = np.linalg.norm(data) * 10
        self.container.append(volume_norm)

    def print_sound(self, search_window=200):
        with sd.Stream(callback=self.check_noise_lvl):
            sd.sleep(search_window)

        self.max_value = max(self.container[-3:])
        #print(f'container -> {self.container}')

        if self.max_value > self.sensitivity:
            return 'user is talking'
        else:
            self.container = []
            return 'user is not talking'


def find_device(p, tags):
    """
    Find an audio device to read input from.
    """
    device_index = None
    for i in range(p.get_device_count()):
        devinfo = p.get_device_info_by_index(i)


        for keyword in tags:
            if keyword in devinfo["name"].lower():

                device_index = i
                return device_index

    if device_index is None:
        print("No preferred input found; using default input device.")

    return device_index


def audio_int(num_samples=50):
    """ Gets average audio intensity of your mic sound. You can use it to get
        average intensities while you're talking and/or silent. The average
        is the avg of the 20% largest intensities recorded.
    """

    print("Getting intensity values from mic.")
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    values = [math.sqrt(abs(audioop.avg(stream.read(CHUNK), 4)))
              for x in range(num_samples)]
    values = sorted(values, reverse=True)
    r = sum(values[:int(num_samples * 0.2)]) / int(num_samples * 0.2)
    print(" Finished ")
    print(" Average audio intensity is ", r)
    stream.close()
    p.terminate()
    return r


def save_audio(wav_file):

    """
    Stream audio from an input device and save it.
    """
    p = pyaudio.PyAudio()

    device = find_device(p, ["input", "mic", "audio"])
    device_info = p.get_device_info_by_index(device)
    channels = int(device_info['maxInputChannels'])

    stream = p.open(
        format=FORMAT,
        channels=channels,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device
    )

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)


    print("* done recording")

    stream.stop_stream()
    stream.close()

    p.terminate()

    wf = wave.open(wav_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def recognize(wav_file):


    test_files = [wav_file]
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]))

    # actual onnx inference and decoding
    onnx_input = input.detach().cpu().numpy()
    ort_inputs = {'input': onnx_input}
    ort_outs = ort_session.run(None, ort_inputs)
    decoded = decoder(torch.Tensor(ort_outs[0])[0])
    return decoded

THRESHOLD = 3000  # The threshold intensity that defines silence
                  # and noise signal (an int. lower than THRESHOLD is silence).

PREV_AUDIO = 0.5  # Previous audio (in seconds) to prepend. When noise
                  # is detected, how much of previously recorded audio is
                  # prepended. This helps to prevent chopping the beggining
                  # of the phrase.

def print_sound(indata):
    volume_norm = np.linalg.norm(indata)*10
    print ("|" * int(volume_norm))

def commatize(num):
    num = [c for c in reversed(str(num))]
    num = [c for c in reversed(list(''.join(l + ',' * (n % 3 == 2) for n, l in enumerate(num))))]
    num = (''.join(num[1:]) if num[0] == ',' else ''.join(num))
    return num

def convert_words_to_num(sent):
    numwords = ['and', 'point',
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety",
        "hundred", "thousand", "million", "billion"
    ]

    split = sent.split()
    num_split = [c if c in numwords else '' for c in split]
    #print(f'num split -> {num_split}')
    word_split = [c if c not in numwords else '' for c in split]
    fixed_words = [c for c in split if c != '']
    #fixed_words = ['/' if c in ['slash', 'slash'] else c for c in fixed_words]
    for i in range(5):
        num_split = [
            c + ' ' + num_split.pop(i + 1) if i < len(num_split) - 1 and c != '' and num_split[i + 1] != '' else c for
            i, c in enumerate(num_split)]
    for i in range(5):
        word_split = [
            c + word_split.pop(i + 1) if i < len(word_split) - 1 and c == '' and word_split[i + 1] == '' else c for i, c
            in enumerate(word_split)]




    if fixed_words != []:

        num_conversion = [word_to_num(word) if word not in ['', ' ', 'and'] else word for word in num_split]
        new_split = [commatize(num_conversion[i]) if num_conversion[i] != '' else c for i, c in enumerate(word_split)]
        #print(new_split)
        new_split = ['/' if c in ['slash', 'slash'] else c for c in new_split]
        new_sent = ' '.join(new_split)
        return new_sent

    else:
        return sent

def listen(filename):
    print('creating a volume container')
    volume_container = VolumeContainer()
    result = '     '
    saved_sents = []
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    nums = '1234567890'
    p = pyaudio.PyAudio()
    device = find_device(p, ["input", "mic", "audio"])
    device_info = p.get_device_info_by_index(device)
    channels = int(device_info['maxInputChannels'])

    stream = p.open(
        format=FORMAT,
        channels=channels,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device
    )

    frames = []
    rel = RATE / CHUNK

    slid_win = deque(maxlen=int(volume_container.silence_time * rel))


    started_talking = False
    count = 0
    usr_talking = False

    print('Begin speaking when ready.')

    print(f'Say "print" to print the current paragraph to {filename} and begin a new paragraph.')
    print(f'Say "threshold" to change the silence threshold. Current silence threshold is {volume_container.silence_threshold}')
    print(f'Say "silence" to change the pause time (How long to wait after speaker is done speaking before printing to screen). Current pause time is {volume_container.silence_time}')
    print()

    while True:

        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)
        slid_win.append(math.sqrt(abs(audioop.avg(data, 4))))
        if (sum([x > volume_container.silence_threshold for x in slid_win]) > 10):
            if started_talking == False:

                started_talking = True

        elif started_talking:

            count += 1
            started_talking = False

            wf = wave.open(f'voiceman_{count % 10}.wav', 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            if usr_talking:
                usr_talking = False
                spoken_sent = ' '.join(saved_sents) + ' ' + recognize(f'voiceman_{count % 10}.wav')
                #saved_sents = []
            else:
                spoken_sent = recognize(f'voiceman_{count % 10}.wav')


            if spoken_sent == 'print':
                with open(filename, 'a', encoding='utf-8') as f:
                    f.write(result + '\n')
                result = '     '
                print()


            elif spoken_sent == 'threshold' or spoken_sent == 'silence threshold':
                try:
                    print('\n\n')
                    volume_container.silence_threshold = int(input(f'     What would you like to set the volume threshold to? Current threshold is {volume_container.silence_threshold}. Please type an integer. ->  '))
                    print(f'new silence threshold -> {volume_container.silence_threshold}')
                except:
                    print('Sorry there was a problem. Please try again.')

            elif spoken_sent == 'silence' or spoken_sent == 'silence time':
                try:
                    print('\n\n')
                    volume_container.silence_time = float(input(f'     What would you like to set the silence time to? Current silence time is {volume_container.silence_time}. Please type an float. ->  '))
                    print(f'new silence time -> {volume_container.silence_time}')

                except:
                    print('Sorry there was a problem. Please try again.')

            elif spoken_sent == 'delete':
                result = result[:result.index(saved_sents[-1]) - 1]
                saved_sents.pop(-1)
                print(f'\r{result}', end='')
                sys.stdout.flush()

            elif not any(item in spoken_sent for item in alphabet):
                pass
            else:

                # post-translational processing of spoken sentence

                spoken_sent = fastpunct.punct(spoken_sent)
                spoken_sent = (', '.join(spoken_sent.split(',')) if ',' in spoken_sent else spoken_sent)
                spoken_sent = convert_words_to_num(spoken_sent)
                spoken_sent = ('. '.join(spoken_sent.split('.')) if '.' in spoken_sent and not any(item in nums for item in spoken_sent) else spoken_sent)


                # last minute check to see if user is talking
                is_usr_talking = volume_container.print_sound(search_window=100)
                #print(f'is_usr_talking -> {is_usr_talking}')

                if is_usr_talking == 'user is talking':
                    usr_talking = True
                    saved_sents.append(spoken_sent)
                else:
                    usr_talking = False
                    saved_sents.append(spoken_sent)
                    result += ' ' + spoken_sent
                    print(f'\r{result}', end='')
                    sys.stdout.flush()

            frames = []

        else:
            if started_talking == True:

                started_talking = False

    p.terminate()


if __name__ == '__main__':
    language = 'en'

    # load required utilities
    _, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language=language)
    (read_batch, split_into_batches,
     read_audio, prepare_model_input) = utils

    # load the actual ONNX model
    try:
        onnx_model = onnx.load('en_v5_xlarge.onnx')
    except:
        onnx_model = torch.hub.download_url_to_file(models.stt_models.en.latest.onnx, 'en_v5_xlarge.onnx', progress=True)

    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession('en_v5_xlarge.onnx')

    while True:

        filename = input('What is the name of the text document you would like to write to?  ->  ')

        print()
        print(f'filename -> {filename}')
        listen(filename)
        save_audio(WAVE_OUTPUT_FILENAME)
