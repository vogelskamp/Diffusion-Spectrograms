import math
import numpy as np
import librosa

SAVE_PATH = "/Users/samvogelskamp/Desktop/Uni/Audio Data Science/data/"
FILE_NAME = "BremerStatmusikanten_-01-padded.npy"

SAVE_PATH2 = "/Users/samvogelskamp/Desktop/Uni/Audio Data Science/padded_audio/"
FILE_NAME2 = "BremerStatmusikanten_-01-padded.wav"

def load_and_convert_to_db(file, sr=None, n_mels=128, time_steps=512):
    data, sr = librosa.load(file, sr=None)

    hop_length = math.floor(len(data)/time_steps)
    start_sample = 0
    length_samples = time_steps * hop_length
    window = data[start_sample:start_sample+length_samples - 1]

    M = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=n_mels, hop_length=hop_length)

    # convert power to db
    return librosa.power_to_db(M, ref=np.max)


def apply_fadeout(audio, sr, duration=0.01):
    # convert to audio indices (samples)
    length = int(duration*sr)
    end = audio.shape[0]
    start = end - length

    # compute fade out curve
    # linear fade
    fade_curve = np.linspace(1.0, 0.0, length)

    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve


def sanity_check():
    SAVED_FILE = "/Users/samvogelskamp/Desktop/Uni/Audio Data Science/data/BremerStatmusikanten_-01-padded.npy"
    OG_FILE = "/Users/samvogelskamp/Desktop/Uni/Audio Data Science/padded_audio/BremerStatmusikanten_-01-padded.wav"

    with open(SAVED_FILE, 'rb') as f:
        arr = np.load(f)

        # convert back to db
        for iy, ix in np.ndindex(arr.shape):
            arr[iy, ix] = arr[iy, ix] * 80 - 80

        data, sr = librosa.load(OG_FILE, sr=None)

        M = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
        M_db = librosa.power_to_db(M, ref=np.max)

        print(M_db[0])
        print(arr[0])