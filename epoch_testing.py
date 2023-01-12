import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import data_transformation.audio_utils as utils

# M_db = utils.load_and_convert_to_db(, n_mels=128, time_steps=512)

with open("S:/Code/_Uni/Diffusion-Spectrograms/results/DDPM_Uncondtional/7.npy", 'rb') as f:
    test = np.load(f)

M_db = []
# convert back to db
for idx in range(0, len(test)):
    M_db.append([])
    for r, g, b in test[idx]:
        M_db[idx].append(r * 80 - 80)

M_db = np.array(M_db)

fig = plt.figure()
fig.set_size_inches(128 / 100, 128 / 100, forward=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

librosa.display.specshow(M_db, ax=ax, cmap='Greys_r')

plt.show()

M = librosa.db_to_power(M_db)
audio = librosa.feature.inverse.mel_to_audio(M, sr=44100, hop_length=1023)

sf.write("result_7.wav", audio, 44100, "PCM_16")