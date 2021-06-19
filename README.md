# LanguageDetection
Used to predict the language from the audio sample. This is constrained to Indian Languages only, but could be extended. 
There are 10 different languages on which the model is trained.
##### Bengali, Hindi, Malayalam, Punjabi, Telugu, Gujarati, Kannada, Marathi, Tamil, Urdu.
The model is trained using 100,000 different samples in every language.

## Installations
Some other packages required - 
```bash
pip install tensorflow
pip install keras
pip install librosa
pip install pydub
pip install moviepy
pip install mutagen
pip install speechrecognition
```
## Usage
Run main.py and input an audio file for classification.

## Training

The model is trained in an iterative fashion. Data is first preprocessed as a flattened (219,13) numpy array and is stored in a text file. 2000 such arrays are stored in every file, which generates 50 files per language.
Then 1 file from every language is taken, and is used for training, in every iteration. Once that iteration is complete, the model is saved, and this model is used
for the next iteration.
This approach is used, since all the data has to be bought into the memory for training, and the data is way over 20 GB, bcz of which this method is implemented.

## Testing
The model is at 87% accuracy. 

## Plots
Since a different approach is used for training, we cannot plot the training from the start.
