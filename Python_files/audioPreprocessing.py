import os
import librosa
import json
import numpy as np
#
def save_mfcc(dataset_path, json_path, no_of_samples, n_mfcc=13, n_fft=2048, hop_length=512, SAMPLE_RATE=22050):
    data = {'mapping':[],'mfcc':[],'labels':[]}
    for i, (dirpath, dirname, filename) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split('/')
            semantic_label = dirpath_components[-1]
            data['mapping'].append(semantic_label)
            print('\nProccessing {}'.format(semantic_label))
            count=0
            for f in filename:
                if count<no_of_samples:
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    mfcc = librosa.feature.mfcc(signal,sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                    mfcc = mfcc.T
                    if len(mfcc) < 218:
                        for _ in range(218-len(mfcc)):
                            mfcc = np.append(mfcc, [mfcc[len(mfcc)-1]], axis=0)
                    data['mfcc'].append(mfcc.tolist())
                    data['labels'].append(i-1)
                    count+=1
                        
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
