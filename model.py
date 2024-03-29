import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

def build_model(input_shape, no_of_classes):
    model = tensorflow.keras.Sequential()

    model.add(tensorflow.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(tensorflow.keras.layers.LSTM(128))

    model.add(tensorflow.keras.layers.Dense(512,activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(0.25))

    model.add(tensorflow.keras.layers.Dense(128,activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(0.25))

    model.add(tensorflow.keras.layers.Dense(32,activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(0.25))

    model.add(tensorflow.keras.layers.Dense(no_of_classes, activation='softmax'))
    return model

def preprocess(language, num):
    array = []
    file = open('Files/Step2/'+language+'/'+language+str(num+1)+'.txt','r')
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
    count=0
    no_of_classes=10
        
    string=''
    for i in range(no_of_classes):
        string+=str(i)
    string = string*2000
    array=list(string)
    y = [int(x) for x in array]
        
    for num in range(12):
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
            model = load_model('model1.h5')
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # model.summary()
        print('Iteration Number -',num)
        history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size = 32, epochs=10)
        model.save('model1.h5')
            
    # plot_history(history)
    test_loss, test_acc = model.evaluate(X_test,y_test,verbose=2)
    print('\nTest accuracy:', test_acc)

if __name__ == '__main__':
    main()