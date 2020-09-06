import os
import librosa
import numpy as np
import subprocess as sp
#
def save_mfcc(dataset_path, no_of_samples, n_mfcc=13, n_fft=2048, hop_length=512, SAMPLE_RATE=22050):    
    for dirpath, dirname, filename in os.walk(dataset_path):
        # if dirpath is not dataset_path:
        dirpath_components = dirpath.split('/')
        language = dirpath_components[-1]
        c1 = 1
        count=0
        for f in filename:
            language_no = 'Files/Step1/'+language+'/'+language+str(c1)
            if count<no_of_samples:
                try:
                    print(language.upper())
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    mfcc = librosa.feature.mfcc(signal,sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                    mfcc = mfcc.T
                    if len(mfcc)>219:
                        for _ in range(len(mfcc)-219):
                            mfcc = np.delete(mfcc, len(mfcc)-1, 0)
                    if len(mfcc) < 219:
                        for _ in range(219-len(mfcc)):
                            mfcc = np.append(mfcc, [mfcc[len(mfcc)-1]], axis=0)
                    mfcc_flatten = np.ndarray.flatten(mfcc)
                    c = mfcc_flatten.shape
                    mfcc_flatten.shape = (c[0],1)
                    mfcc_flatten = mfcc_flatten.T
                    with open(language_no+'.txt','a') as f:
                        np.savetxt(f, mfcc_flatten, fmt='%.3f')
                    count+=1
                except:
                    continue
            else:
                count=0
                c1+=1
# save_mfcc('AugmentedDatasets/Bengali',no_of_samples=2000)
# save_mfcc('AugmentedDatasets/Gujarati', no_of_samples=2000)
save_mfcc('AugmentedDatasets/Malayalam', no_of_samples=2000)
# save_mfcc('AugmentedDatasets/Hindi', no_of_samples=2000)
# save_mfcc('AugmentedDatasets/Kannada', no_of_samples=2000)
# save_mfcc('AugmentedDatasets/Tamil', no_of_samples=2000)
# save_mfcc('AugmentedDatasets/Telugu', no_of_samples=2000)
# save_mfcc('AugmentedDatasets/Marathi',no_of_samples=2000)
# save_mfcc('AugmentedDatasets/Urdu',no_of_samples=2000)
# save_mfcc('AugmentedDatasets/Punjabi',no_of_samples=2000)



# dataset_path='AugmentedDatasets/Gujarati'
# no_of_samples=6
# n_mfcc=13
# n_fft=2048
# hop_length=512
# SAMPLE_RATE=22050
# count_2=0

# with open('Files/Step1/Kannada/Kannada1.txt','r') as f:
#     for i in range(100):
#         line = f.readline()
#         line = line.split()
#         print(len(line))
# import time
# for j in range(4,50):
#     for i in range(90):
#         print('File-',j,' ',(sp.getoutput('wc -l Files/Kannada/Kannada1.txt')).split()[0])
#         time.sleep(10)

