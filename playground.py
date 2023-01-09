import numpy as np
import librosa
import soundfile as sf
import data_transformation.audio_utils as utils
 
M_db = utils.load_and_convert_to_db("C:/Users/student-isave/Documents/Diffusion-Spectrograms/padded_audio/Frau_DeKock_DieEhemänner-01-padded.wav", n_mels=128, time_steps=512)

# with open('C:/Users/student-isave/Documents/Diffusion-Spectrograms/audio_data/Frau_DeKock_DieEhemänner-01-padded.npy', 'rb') as f:
#     test = np.load(f)

M = librosa.db_to_power(M_db)
audio = librosa.feature.inverse.mel_to_audio(M, sr=44100, hop_length=1023)

print(M_db.shape)
sf.write("test.wav", audio, 44100, "PCM_16")