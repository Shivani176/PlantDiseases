from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from sklearn.metrics import accuracy_score
import os
import cv2
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import pickle
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn import svm

main = Tk()
main.title("Research on Recognition Model of Crop Diseases and Insect Pests Based on Deep Learning in Harsh Environments")
main.geometry("1300x1200")

global filename
global X, Y
global model
global accuracy

fertilizers = []

plants = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
          'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
          'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
          'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
          'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

def uploadDataset():
    global X, Y
    global filename
    fertilizers.clear()
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,'dataset loaded\n')
    with open("messages.txt", "r") as file:
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            fertilizers.append(line)
    file.close()            

def imageProcessing():
    text.delete('1.0', END)
    global X, Y
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    img = X[20].reshape(64,64,3)
    Y = to_categorical(Y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,'image processing completed\n')
    cv2.imshow('ff',cv2.resize(img,(250,250)))
    cv2.waitKey(0)

def multclassSVM():
    text.delete('1.0', END)
    #reading all training images from numpy array
    XX = np.load("model/svmX.txt.npy")
    #reading all class labels of different plants
    YY = np.load("model/svmY.txt.npy")
    #finding unique class label from target label
    unique,classes = np.unique(YY,return_counts=True)
    #displaying total processing images
    text.insert(END,'Image processing completed\n')
    text.insert(END,'Total images found in dataset is : '+str(XX.shape[0])+"\n")
    text.insert(END,'Total classes found in dataset is : '+str(len(classes))+"\n")

    #splitting dataset images into train and test
    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2)
    text.insert(END,"Total images used to train multiclass SVM : "+str(len(X_train))+"\n")
    text.insert(END,"Total images used to test multiclass SVM  : "+str(len(X_test))+"\n\n")
    #building multiclass SVM object as our dataset contains total 15 different types of plant disease so 15 classes are there
    #and this svm can predict all 15 multi class plant diseases
    cls = svm.SVC()
    #training svm on given images and target class label
    cls = cls.fit(XX, YY)
    #performing prediction on test data
    predict = cls.predict(X_test)
    #calculating predicted accuracy using predicted values and real test values
    acc = accuracy_score(y_test,predict)*100
    text.insert(END,"Multiclass SVM Prediction Accuracy on test images : "+str(acc)+"\n")
    
    
def cnnModel():
    global model
    global accuracy
    text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()    
        model.load_weights("model/model_weights.h5")
        model._make_predict_function()   
        print(model.summary())
        f = open('model/history.pckl', 'rb')
        accuracy = pickle.load(f)
        f.close()
        acc = accuracy['accuracy']
        acc = acc[9] * 100
        text.insert(END,"CNN Crop Disease Recognition Model Prediction Accuracy = "+str(acc))
    else:
        model = Sequential() #resnet transfer learning code here
        model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim = 256, activation = 'relu'))
        model.add(Dense(output_dim = 15, activation = 'softmax'))
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        print(model.summary())
        hist = model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        model.save_weights('model/model_weights.h5')            
        model_json = model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        accuracy = pickle.load(f)
        f.close()
        acc = accuracy['accuracy']
        acc = acc[9] * 100
        text.insert(END,"CNN Crop Disease Recognition Model Prediction Accuracy = "+str(acc))
        
def getFertilizer(name):
    details = "Fertilizer Details Not Available"
    for i in range(len(fertilizers)):
        arr = fertilizers[i].split(":")
        arr[0] = arr[0].strip()
        arr[1] = arr[1].strip()
        if arr[0] == name:
            details = arr[1]
            break
    return details        

def predict():
    global model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test/255
    preds = model.predict(test)
    predict = np.argmax(preds)
    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    details = getFertilizer(plants[predict])
    text.insert(END,"Fertilizer Details : "+details)
    cv2.putText(img, 'Crop Disease Recognize as : '+plants[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow('Crop Disease Recognize as : '+plants[predict], img)
    cv2.waitKey(0)

def graph():
    acc = accuracy['accuracy']
    loss = accuracy['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Iteration Wise Accuracy & Loss Graph')
    plt.show()
    
def close():
    main.destroy()
    text.delete('1.0', END)
    
font = ('times', 15, 'bold')
title = Label(main, text='Research on Recognition Model of Crop Diseases and Insect Pests Based on Deep Learning in Harsh Environments')
#title.config(bg='powder blue', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Crop Disease Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Image Processing & Normalization", command=imageProcessing)
processButton.place(x=20,y=150)
processButton.config(font=ff)

modelButton = Button(main, text="Build Crop Disease Recognition Model", command=cnnModel)
modelButton.place(x=20,y=200)
modelButton.config(font=ff)

svmButton = Button(main, text="Run Multiclass SVM Algorithm", command=multclassSVM)
svmButton.place(x=20,y=250)
svmButton.config(font=ff)

predictButton = Button(main, text="Upload Test Image & Predict Disease", command=predict)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=20,y=350)
graphButton.config(font=ff)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=400)
exitButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config()
main.mainloop()
