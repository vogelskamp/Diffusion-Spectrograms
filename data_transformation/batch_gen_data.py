import math
import os
import tkinter.filedialog

import audio_utils as utils
import librosa
import numpy as np
import soundfile as sf
from augment import augment
from tqdm import tqdm

MAX_LENGTH = 10
TIME_STEPS = 512
N_MELS = 128

ORIGIN_FOLDER_PATH = tkinter.filedialog.askdirectory(
    title='Select Origin Folder')

DESTINATION_FOLDER_PATH = tkinter.filedialog.askdirectory(
    title='Select Destination Folder') + "/"

for base, dirs, files in os.walk(ORIGIN_FOLDER_PATH):

    print('Augmenting: ', base)
    for file in tqdm(files):
        # Append Filepath to current Filepath
        currentFilepath = ORIGIN_FOLDER_PATH + "/" + file

        # load File
        data, sr = librosa.load(currentFilepath, sr=None)

        if len(data) > MAX_LENGTH * sr:
            print(file, " is too long, skipping...")
            continue

        # add padding and fadeout
        utils.apply_fadeout(data, sr)

        TARGET_SAMPLES = sr * MAX_LENGTH
        data = np.append(data, np.zeros(TARGET_SAMPLES - len(data)))

        # expand data through augmentation
        expanded_data = augment(data, sr, file.split('.')[-2])

        # generate mel spectrograms
        for audio, name in expanded_data:
            hop_length = math.floor(len(data)/TIME_STEPS)
            start_sample = 0
            length_samples = TIME_STEPS * hop_length
            window = audio[start_sample:start_sample+length_samples - 1]

            M = librosa.feature.melspectrogram(
                y=window, sr=sr, n_mels=N_MELS, hop_length=hop_length)

            # convert power to db
            M_db = librosa.power_to_db(M, ref=np.max)

            # convert 0 to -80 -> 1 to 0
            for iy, ix in np.ndindex(M_db.shape):
                M_db[iy, ix] = 1 - M_db[iy, ix] / -80

            # convert to 3D array
            M_db = M_db[np.newaxis, :, :]

            # save
            NEW_NAME = DESTINATION_FOLDER_PATH + name + ".npy"
            with open(NEW_NAME, 'wb') as f:
                # save to file
                np.save(f, M_db)
