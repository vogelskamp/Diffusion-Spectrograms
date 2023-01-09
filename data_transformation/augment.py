import librosa
import numpy as np
import soundfile as sf
import tkinter.filedialog
import os

ORIGIN_FOLDER_PATH = tkinter.filedialog.askdirectory(title='Select Origin Folder')

DESTINATION_FOLDER_PATH = tkinter.filedialog.askdirectory(title='Select Destination Folder') + "/"

print(DESTINATION_FOLDER_PATH)

for base, dirs, files in os.walk(ORIGIN_FOLDER_PATH):
    print('Augmenting: ',base)
        
    for Files in files:
       
       #Append Filepath to current Filepath
       currentFilepath = ORIGIN_FOLDER_PATH + "/" + Files
       
       #load File 
       currentfile, sr = librosa.load(currentFilepath, sr=None)
      
       #Incrase Tempo by 1, 2, 3, 4, 5%
       augmentdata = librosa.effects.time_stretch(currentfile, 1.01)
       sf.write(DESTINATION_FOLDER_PATH + "f1_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.time_stretch(currentfile, 1.02)
       sf.write(DESTINATION_FOLDER_PATH + "f2_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.time_stretch(currentfile, 1.03)
       sf.write(DESTINATION_FOLDER_PATH + "f3_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.time_stretch(currentfile, 1.04)
       sf.write(DESTINATION_FOLDER_PATH + "f4_"+Files,augmentdata, sr)
       
       augmentdata = librosa.effects.time_stretch(currentfile, 1.05)
       sf.write(DESTINATION_FOLDER_PATH + "f5_"+Files,augmentdata, sr)
       
       #Decrease Tempo by 1, 2, 3, 4, 5%
       augmentdata = librosa.effects.time_stretch(currentfile, 0.99)
       sf.write(DESTINATION_FOLDER_PATH + "s1_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.time_stretch(currentfile, 0.98)
       sf.write(DESTINATION_FOLDER_PATH + "s2_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.time_stretch(currentfile, 0.97)
       sf.write(DESTINATION_FOLDER_PATH + "s3_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.time_stretch(currentfile, 0.96)
       sf.write(DESTINATION_FOLDER_PATH + "s4_"+Files,augmentdata, sr)
       
       augmentdata = librosa.effects.time_stretch(currentfile, 0.95)
       sf.write(DESTINATION_FOLDER_PATH + "s5_"+Files,augmentdata, sr)

       #Pitch Shift Up 1, 2, 3, ,4 , 5 Semitones
       augmentdata = librosa.effects.pitch_shift(currentfile, sr, 1)
       sf.write(DESTINATION_FOLDER_PATH + "pu1_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.pitch_shift(currentfile, sr, 2)
       sf.write(DESTINATION_FOLDER_PATH + "pu2_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.pitch_shift(currentfile, sr, 3)
       sf.write(DESTINATION_FOLDER_PATH + "pu3_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.pitch_shift(currentfile, sr, 4)
       sf.write(DESTINATION_FOLDER_PATH + "pu4_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.pitch_shift(currentfile, sr, 5)
       sf.write(DESTINATION_FOLDER_PATH + "pu5_"+Files,augmentdata, sr)

       #Pitch Shift Down 1, 2, 3, ,4 , 5 Semitones
       augmentdata = librosa.effects.pitch_shift(currentfile, sr, 1)
       sf.write(DESTINATION_FOLDER_PATH + "pd1_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.pitch_shift(currentfile, sr, 2)
       sf.write(DESTINATION_FOLDER_PATH + "pd2_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.pitch_shift(currentfile, sr, 3)
       sf.write(DESTINATION_FOLDER_PATH + "pd3_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.pitch_shift(currentfile, sr, 4)
       sf.write(DESTINATION_FOLDER_PATH + "pd4_"+Files,augmentdata, sr)

       augmentdata = librosa.effects.pitch_shift(currentfile, sr, 5)
       sf.write(DESTINATION_FOLDER_PATH + "pd5_"+Files,augmentdata, sr)

       #print Info
       print( Files + "is done")
