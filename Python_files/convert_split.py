import os
from mutagen.mp3 import MP3
from pydub import AudioSegment
import datetime
import librosa
from soundfile import write
import numpy as np
import random
#
'''Converts .wav files into .mp3 and splits the files according to the time 
interval specified and store in their respective output folders'''

def wavetomp3(input_folder, output_folder):
    for i,j,k in os.walk(input_folder):
        files = k
    for i,j in enumerate(files):
        try:
            export_file = os.path.join(output_folder,'file')+str(i)+'.mp3'
            AudioSegment.from_wav(os.path.join(input_folder,j)).export(export_file, format="mp3")
        except:
            continue
# wavetomp3('Datasets_wav/Kannada','Kannada')
    
def split(input_folder, output_folder, time_interval=5):
    for i,j,k in os.walk(input_folder):
        files = k
    k1=0
    for j in files:
        try:
            input_file = os.path.join(input_folder,j)
            audio = MP3(input_file)
            audio_length = audio.info.length
            start_time = datetime.datetime(100,1,1,0,0,0)
            for _ in range(int(audio_length)//time_interval):
                output_file = os.path.join(output_folder,str(k1))+'.mp3'
                finish_time = start_time + datetime.timedelta(seconds=time_interval)
                command = 'ffmpeg -i '+input_file+' -acodec copy -ss '+str(start_time).split(' ')[1]+' -to '+str(finish_time).split(' ')[1]+' '+output_file    
                os.system(command)
                start_time = finish_time
                k1+=1
        except:
            continue
        

# split('Datasets_mp3/Gujarati','Datasets/Gujarati')
# split('Datasets_mp3/Kannada','Datasets/Kannada')
# split('Datasets_mp3/Telugu','Datasets/Telugu')
# split('Datasets_mp3/Tamil','Datasets/Tamil')
# split('Datasets_mp3/Malayalam','Datasets/Malayalam')
# split('/home/chaitanya2923/ab','Datasets/Hindi')


def augmentation(input_folder, output_folder):
    array = np.linspace(0.01, 0.1,100)
    steps = list(map(int,np.linspace(-3,-8, 6)))
    roll = list(map(int,np.linspace(5,20,4)))
    
    for i,j,k in os.walk(input_folder):
        files = k
    for j in files:
        try:
            input_file = os.path.join(input_folder,j)
            signal, sr = librosa.load(input_file, sr=22050)
            
            # 1
            r_num = round(random.choice(array), 3)
            signal_n = signal + r_num*np.random.normal(0,1,len(signal))
            
            output_file_n_wav = os.path.join(output_folder,j[:-4])+'_n.wav'
            output_file_n_mp3 = os.path.join(output_folder,j[:-4])+'_n.mp3'
            write(output_file_n_wav,signal_n,sr)
            AudioSegment.from_wav(output_file_n_wav).export(output_file_n_mp3, format="mp3")
            command = 'rm '+output_file_n_wav
            os.system(command)
            
            # 2
            signal_roll = np.roll(signal,int(sr/random.choice(roll)))
            
            signal_pitch_sf = librosa.effects.pitch_shift(signal_roll,sr,n_steps=random.choice(steps))
            
            output_file_pitch_wav = os.path.join(output_folder,j[:-4])+'_pitch.wav'
            output_file_pitch_mp3 = os.path.join(output_folder,j[:-4])+'_pitch.mp3'
            write(output_file_pitch_wav,signal_pitch_sf,sr)
            AudioSegment.from_wav(output_file_pitch_wav).export(output_file_pitch_mp3, format="mp3")
            command = 'rm '+output_file_pitch_wav
            os.system(command)
            
            # 3
            signal_time_stch = librosa.effects.time_stretch(signal,0.8)[:int(signal.shape[0]/22050)]            
        
            output_file_time_wav = os.path.join(output_folder,j[:-4])+'_time.wav'
            output_file_time_mp3 = os.path.join(output_folder,j[:-4])+'_time.mp3'
            write(output_file_time_wav,signal_time_stch,sr)
            AudioSegment.from_wav(output_file_time_wav).export(output_file_time_mp3, format="mp3")
            command = 'rm '+output_file_time_wav
            os.system(command)
        
        except:
            continue

# augmentation('Datasets/Gujarati','AugmentedDatasets/Gujarati')
# augmentation('Datasets/Hindi','AugmentedDatasets/Hindi')
# augmentation('Datasets/Telugu','AugmentedDatasets/Telugu')
# augmentation('Datasets/Kannada','AugmentedDatasets/Kannada')
# augmentation('Datasets/Tamil','AugmentedDatasets/Tamil')
# augmentation('Datasets/Malayalam','AugmentedDatasets/Malayalam')
