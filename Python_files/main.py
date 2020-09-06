import os
from keras.models import load_model
import numpy as np
import speech_recognition as sr
import librosa
from mutagen.mp3 import MP3
from pydub import AudioSegment
from convert_split import wavetomp3, split
# from mfcc_text import save_mfcc
#
# Record user audio
def record():
    mic = sr.Microphone()
    recog = sr.Recognizer()
    
    with mic as source:
        print("Say now!!!!")
        recog.adjust_for_ambient_noise(source)
        audio = recog.listen(source)
        with open('audiofile.wav','wb') as a:
            a.write(audio.get_wav_data())

# def save(ll):
#     n_mfcc=13
#     n_fft=2048
#     hop_length=512
#     count_2=0
#     temp_array = []
#     file_path = 'Dummy/output2/A/'

#     for i in ll:
#         file = os.path.join(file_path,i)
#         signal, sr = librosa.load(file, sr=22050)
#         mfcc = librosa.feature.mfcc(signal,sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
#         mfcc = mfcc.T
#         if len(mfcc) < 219:
#             for _ in range(219-len(mfcc)):
#                 mfcc = np.append(mfcc, [mfcc[len(mfcc)-1]], axis=0)
#         mfcc_flatten = np.ndarray.flatten(mfcc)
#         mfcc_flatten = mfcc_flatten.tolist()
#         temp_array+=mfcc_flatten
#         temp_array_np = np.array(temp_array)
#         c = temp_array_np.shape
#         temp_array_np.shape = (c[0],1)
#         temp_array_np = temp_array_np.T
#         count_2+=1
#         with open('A.txt','a') as f:
#             np.savetxt(f, temp_array_np, fmt='%.3f')
#             count_2=0
#             temp_array = []
        
#Remove Dummy, if exists and create different temp folders to store files gem=nerated.
if os.path.exists('Dummy'):
    os.system('rm -r Dummy')
os.system('mkdir Dummy')
os.system('mkdir Dummy/input')
os.system('mkdir Dummy/output')
os.system('mkdir Dummy/output2')

input_folder = 'Dummy/input'
output_folder = 'Dummy/output'
output_folder2 = 'Dummy/output2'

A='mkdir '+os.path.join(output_folder2,'A')
os.system(A)

#load the model
model = load_model('model.h5')
label = {0:'Bengali',1:'Gujarati',2:'Hindi',3:'Kannada',4:'Malayalam',5:'Marathi',6:'Punjabi',7:'Tamil',8:'Telugu',9:'Urdu'}
i = int(input('Do you wanna speak? (1/0)\n'))
if i:
    record()
    os.system('ffmpeg -i audiofile.wav audiofile.mp3')
    input_file = 'audiofile.mp3'
else:
    input_file = input('Input File:')

sound = AudioSegment.from_mp3(input_file)
audio = MP3(input_file)
audio_length = audio.info.length
if audio_length<5:
    silence = AudioSegment.silent(duration=5000)
    sound += silence
    sound.export('audiofile.mp3',format='mp3')
    input_file = 'audiofile.mp3'
    
#copy input file to input folder for further usage
command = "cp "+input_file+" "+input_folder
os.system(command)

#check wheather audio is of wav or mp3 format
if input_file[-4:] == '.wav':
    wavetomp3(input_folder, output_folder)
split(input_folder, output_folder2)

ll = os.listdir(output_folder2)
ll = [x for x in ll if x[-4:]=='.mp3']
for l in ll:
    ff = 'mv '+os.path.join(output_folder2,l)+' '+os.path.join(output_folder2,'A')
    os.system(ff)

# Generation of MFCC text file
# save(ll)
n_mfcc=13
n_fft=2048
hop_length=512
count_2=0
temp_array = []
file_path = 'Dummy/output2/A/'

for i in ll:
    file = os.path.join(file_path,i)
    signal, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(signal,sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc = mfcc.T
    if len(mfcc) < 219:
        for _ in range(219-len(mfcc)):
            mfcc = np.append(mfcc, [mfcc[len(mfcc)-1]], axis=0)
    mfcc_flatten = np.ndarray.flatten(mfcc)
    mfcc_flatten = mfcc_flatten.tolist()
    temp_array+=mfcc_flatten
    temp_array_np = np.array(temp_array)
    c = temp_array_np.shape
    temp_array_np.shape = (c[0],1)
    temp_array_np = temp_array_np.T
    count_2+=1
    with open('A.txt','a') as f:
        np.savetxt(f, temp_array_np, fmt='%.3f')
        count_2=0
        temp_array = []

# Loading the json file
with open('A.txt','r') as f:
    X = []
    lines = f.readlines()
    for line in lines:
        split = line.split()
        split = [float(x) for x in split]
        split = np.array(split)
        X.append(np.reshape(split,(219,13)))
X = np.array(X)
print(X.shape)
# Prediction Phase
# languages, countarray, val = [],[],[]
print()
for i in X:
    i.shape=(1,219,13)
    D = model.predict(i).tolist()
    D = D[0]
    print('\n',label[D.index(max(D))],'with {0:.1f}% confidence'.format(max(D)*100))
    # languages.append(D.index(max(D)))
    # print(D)
# print(languages)

# for i in range(len(label)):
#     countarray.append(languages.count(i))
# print(countarray)
# print('\n'+label[countarray.index(max(countarray))]+', with {0:.1f}% confidence'.format(max(D)*100))

os.system('rm -r Dummy')
os.system('rm A.txt')


