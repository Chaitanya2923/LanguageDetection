import os
from convert_split import wavetomp3, split, augmentation
from convert_split import augmentation
import mfcc_image
import CNN
import shutil
languages = ['Telugu', 'Kannada', 'Tamil', 'Malayalam', 'Gujarati', 'Hindi']
#
if not os.path.exists('Datasets_mp3'):
    os.system('mkdir Datasets_mp3')
if not os.path.exists('Datasets'):
    os.system('mkdir Datasets')
if not os.path.exists('AugmentedDataset'):
    os.system('mkdir AugmentedDataset')  
if not os.path.exists('ImageDataset'):
    os.system('mkdir ImageDataset')
    os.system('mkdir ImageDataset/Train')
    os.system('mkdir ImageDataset/Test')
    
# Load wav files under its corresponding language folder.
def dataPreparation(language):
    input_folder_wave = os.path.join('Datasets_wav',language)
    output_folder_mp3 = os.path.join('Datasets_mp3',language)
    output_folder_split = os.path.join('Datasets',language)
    aug_folder = os.path.join('AugmentedDataset',language)

    command = 'mkdir Datasets_mp3/'+language
    os.system(command)
    command = 'mkdir Datasets/'+language
    os.system(command)
    command = 'mkdir AugmentedDataset/'+language
    os.system(command)
    command = 'mkdir ImageDataset/Train/'+language
    os.system(command)
    command = 'mkdir ImageDataset/Test/'+language
    os.system(command)
    
    # wavetomp3(input_folder_wave, output_folder_mp3)
    # split(output_folder_mp3, output_folder_split, time_interval=5)
    augmentation(output_folder_split, aug_folder)
    
    source = 'Datasets/'+'Hindi'+'/'
    dest1 = 'AugmentedDataset/'+'Hindi'
    files = os.listdir(source)
    for f in files:
        shutil.move(source+f, dest1)
    
def train_test_split(language):
    length = len(os.listdir('ImageDataset/Train/'+language))
    percentage = int(length*(20/100))   
    for k in range(0+9730,percentage+9730):
        source = 'ImageDataset/Train/'+language+'/'+str(k)+'.png'
        destination = 'ImageDataset/Test/'+language+'/'
        command = 'mv'+' '+ source+' ' + destination
        os.system(command)

if __name__ == '__main__':
    no_of_classes = 6
    c_array = []
    dataset_path = 'AugmentedDatasets'
    for language in languages:
        dataPreparation(language)
        c_array.append(len(os.listdir('D/'+language)))
    no_of_samples = min(c_array)
    mfcc_image.save_mfcc(dataset_path, no_of_samples=6000)
    for language in languages:
        train_test_split(language)
    CNN.main('ImageDataset', no_of_classes)
    