import librosa
import librosa.display
import numpy as np
import soundfile as sf
import math
import os
from tqdm import tqdm
import audio_utils as utils

FILE_PATH = "C:/Users/student-isave/Documents/Diffusion-Spectrograms/padded_audio/"
SAVE_PATH = "C:/Users/student-isave/Documents/Diffusion-Spectrograms/audio_data/"

n_mels = 128  # number of bins in spectrogram. Height of image

for file in tqdm(os.listdir(FILE_PATH)):

    if not file.endswith("-padded.wav"):
        continue

    M_db = utils.load_and_convert_to_db(FILE_PATH + file, n_mels=n_mels, time_steps=512)

    # convert 0 to -80 -> 1 to 0
    for iy, ix in np.ndindex(M_db.shape):
        M_db[iy, ix] = 1 - M_db[iy, ix] / -80

    NEW_NAME = SAVE_PATH + file.replace(".wav", ".npy")
    with open(NEW_NAME, 'wb') as f:
        # save to file
        np.save(f, M_db)

