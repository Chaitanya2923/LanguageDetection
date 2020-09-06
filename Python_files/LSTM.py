import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import subprocess as sp
#adding commentt
def build_model(input_shape, no_of_classes):
    model = keras.Sequential()

    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(128))

    model.add(keras.layers.Dense(512,activation='relu'))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Dense(32,activation='relu'))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Dense(no_of_classes, activation='softmax'))
    return model

# def plot_history(history):
# 	fig,axs=plt.subplots(2)
# 	axs[0].plot(history.history['accuracy'], label='train_accuracy')
# 	axs[0].plot(history.history['val_accuracy'], label='test_accuracy')
# 	axs[0].set_xlabel('Epoch')
# 	axs[0].set_ylabel('Accuracy')
# 	axs[0].legend(loc='lower_right')
# 	axs[0].set_title('Accuracy_eval')

# 	axs[1].plot(history.history['loss'], label='train_error')
# 	axs[1].plot(history.history['val_loss'], label='test_error')
# 	axs[0].set_xlabel('Epoch')
# 	axs[1].set_ylabel('Error')
# 	axs[1].legend(loc='upper_right')
# 	axs[1].set_title('Error_eval')

def preprocess(language, num):
    array = []
    file = open('Files/'+language+'/'+language+str(num)+'.txt','r')
    lines = file.readlines()
    for line in lines:
        split = line.split()
        split = [float(x) for x in split]
        array.append(split)
    final_array = []
    array = np.array(array)
    for i in range(len(array)):
        final_array.append(np.reshape(array[i], (219,13)))
    final_array = np.array(final_array)
    return final_array

def main():    
    count=1
    no_of_classes=10
        
    string=''
    for i in range(no_of_classes):
        string+=str(i)
    string = string*2000
    array=list(string)
    y = [int(x) for x in array]
        
    for num in range(31,51):
        X = []
        bengali = preprocess('Bengali',num)
        gujarati = preprocess('Gujarati',num)
        hindi = preprocess('Hindi',num)
        kannada = preprocess('Kannada',num)
        malayalam = preprocess('Malayalam',num)
        marathi = preprocess('Marathi',num)
        punjabi = preprocess('Punjabi',num)
        tamil = preprocess('Tamil',num)
        telugu = preprocess('Telugu',num)
        urdu = preprocess('Urdu',num)
        
        for i in range(2000):
            X.append(bengali[i])
            X.append(gujarati[i])
            X.append(hindi[i])
            X.append(kannada[i])
            X.append(malayalam[i])
            X.append(marathi[i])
            X.append(punjabi[i])
            X.append(tamil[i])
            X.append(telugu[i])
            X.append(urdu[i])
        X = np.array(X)
        
        del [bengali,gujarati,hindi,kannada,malayalam,marathi,punjabi,tamil,telugu,urdu]
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size=0.2)
        X_train, X_test,X_validation = np.array(X_train), np.array(X_test), np.array(X_validation)
        y_train, y_test,y_validation = np.array(y_train), np.array(y_test), np.array(y_validation)
        
        del X
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        if count == 0:
            model = build_model(input_shape,no_of_classes)
            count+=1
        else:
            model = load_model('model.h5')
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # model.summary()
        print('Iteration Number -',num)
        history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size = 32, epochs=10)
        model.save('model.h5')
            
    # plot_history(history)
    test_loss, test_acc = model.evaluate(X_test,y_test,verbose=2)
    print('\nTest accuracy:', test_acc)
main()

# int((sp.getoutput('wc -l Gujarati.txt')).split()[0])