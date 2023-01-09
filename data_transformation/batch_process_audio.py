import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import audio_utils as utils

FILE_PATH = "C:/Users/student-isave/Documents/Diffusion-Spectrograms/audio/"
# FILE_NAME = "BremerStatmusikanten_-05.wav"

for file in tqdm(os.listdir(FILE_PATH)):

    if file.endswith("-padded.wav"):
        print(file, "does not end in -padded.wav, ignoring..")
        continue

    y, sr = librosa.load(FILE_PATH + file, sr=None)

    utils.apply_fadeout(y, sr)

    TARGET_SAMPlES = 524000

    y = np.append(y, np.zeros(TARGET_SAMPlES - len(y)))

    NEW_NAME = FILE_PATH + "../padded_audio/" + \
        file.replace(".wav", "") + "-padded.wav"
    sf.write(NEW_NAME, y, sr, 'PCM_16')