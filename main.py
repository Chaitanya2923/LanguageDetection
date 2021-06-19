import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model
import numpy as np
import librosa
from mutagen.mp3 import MP3
from pydub import AudioSegment
import itertools
from tabulate import tabulate
from collections import Counter
import datetime
import shutil

#Remove Dummy, if exists and create different temp folders to store files gem=nerated.
def createFolders():
    if os.path.exists('Dummy'):
        shutil.rmtree('Dummy')
    os.mkdir('Dummy')
    os.mkdir('Dummy/input')
    os.mkdir('Dummy/output')
    os.mkdir('Dummy/output2')
    os.mkdir('Dummy/output2/A')

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

def processInput():
    input_folder = 'Dummy/input'
    output_folder = 'Dummy/output'
    output_folder2 = 'Dummy/output2'
    
    input_file = input('Input File:\n')
    # input_file = input()
    actual_language = input_file.split('\\')[-2]
    
    sound = AudioSegment.from_mp3(input_file)
    audio = MP3(input_file)
    audio_length = audio.info.length
    if audio_length<5:
        silence = AudioSegment.silent(duration=5000)
        sound += silence
        sound.export('audiofile.mp3',format='mp3')
        input_file = 'audiofile.mp3'
        
    #copy input file to input folder for further usage
    shutil.copy(input_file, input_folder)
    
    split(input_folder, output_folder2)
    # augmentation(output_folder, output_folder2)
    
    splitFiles = [x for x in os.listdir(output_folder2) if x[-4:]=='.mp3' and 'time' not in x]
    for l in splitFiles:
        shutil.move(os.path.join(output_folder2,l), os.path.join(output_folder2,'A'))
    return splitFiles, audio_length, actual_language

def save(splitFiles):
    n_mfcc=13
    n_fft=2048
    hop_length=512
    count_2=0
    temp_array = []
    file_path = 'Dummy/output2/A'
    
    for i in splitFiles:
        file = os.path.join(file_path,i)
        signal, sr = librosa.load(file, sr=22050)
        mfcc = librosa.feature.mfcc(signal,sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) < 219:
            for _ in range(219-len(mfcc)):
                mfcc = np.append(mfcc, [mfcc[len(mfcc)-1]], axis=0)
        elif len(mfcc)>219:
            for _ in range(len(mfcc)-219):
                mfcc = np.delete(mfcc, len(mfcc)-1, 0)
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

# Loading the text file
def read_output_file():
    with open('A.txt','r') as f:
        X = []
        lines = f.readlines()
        for line in lines:
            split = line.split()
            split = [float(x) for x in split]
            split = np.array(split)
            X.append(np.reshape(split,(219,13)))
    os.remove('A.txt')
    X = np.array(X)
    print(X.shape)
    return X

def predict(X, audio_length):
    model = load_model('Models/model.h5')
    label = {0:'Bengali',1:'Gujarati',2:'Hindi',3:'Kannada',4:'Malayalam',5:'Marathi',6:'Punjabi',7:'Tamil',8:'Telugu',9:'Urdu'}
    # M = np.zeros(10)
    final_lang_array = []
    count = 0
    tab_str1 = 'Audio length: '+str(round(audio_length,2))+' sec\n\n'
    tab_str2 = ''
    for i in X:
        i.shape=(1,219,13)
        D = model.predict(i).tolist()
        D = D[0]
        final_dict = {}
        for ii in range(10):
            final_dict[label[ii]] = round(D[ii]*100,1)
            
        temp_f = {k:v for k,v in sorted(final_dict.items(), key=lambda item: item[1], reverse=True)}
        out = dict(itertools.islice(temp_f.items(), 3))
        
        tab_str1+='In '+str(count)+' to '+str(count+5)+' sec interval:\n'
        for i in out.items():
            tab_str1+=str(i[0])+' - '+str(i[1])+'%\n'
        tab_str1+='\n'            
        final_lang_array.append(tuple(out.items())[0][0])
        count+=5
    return final_lang_array, tab_str1, tab_str2


def output(final_lang_array, tab_str1, tab_str2, actual_language):
    print(tabulate([[tab_str1]], tablefmt='fancy_grid'))   

    d = Counter(final_lang_array)
    if all(val == 1 for val in d.values()):
    	language = ', '.join(final_lang_array)
    else:
    	language = max(d, key=d.get)

    print(tabulate([['Actual Language: '+actual_language+'\n'+'Final prediction: '+language]], tablefmt='fancy_grid'))

def mainn():
    createFolders()
    splitFiles, audio_length, actual_language = processInput()
    save(splitFiles)
    shutil.rmtree('Dummy')
    # os.remove('AudioFiles/audiofile.mp3')
    X = read_output_file()
    final_lang_array, tab_str1, tab_str2 = predict(X, audio_length)
    output(final_lang_array, tab_str1, tab_str2, actual_language)
    
if __name__ == '__main__':
    mainn()