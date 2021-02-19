# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:44:37 2021

@author: HP
"""

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from time import sleep


def main():
   '''Pobieranie zestawu danych''' 
   mnist=tf.keras.datasets.fashion_mnist
   (x_train, y_train), (x_test, y_test)=mnist.load_data()
   
   while(True):
       option=input("1.Zbuduj i wyszkol siec\n2.Pokaz obraz\n3.Klasyfikacja\n4.Wyjscie\n")
       if option=='1':
           model=Model(x_train, y_train, x_test, y_test)
       elif(option=='2'):
           nr=input("Podaj numer obrazu do wyswietlenia\n")
           Show_img(x_test/255.0, int(nr))
       elif option=='3':
            nr=input("Podaj numer obrazu do klasyfikacji\n")
            img=(numpy.expand_dims(x_test[int(nr)]/255.0, 0))
            Clasificate(model, img, x_test[int(nr)])
       elif option=='4':
           break
       sleep(0.05)
   
   
def Show_img(x_train, nr=0):   
   plt.figure()
   plt.imshow(x_train[nr])
   plt.colorbar()
   plt.grid(False)
   plt.show()
   
def Model(x_train, y_train, x_test, y_test):
   x_train, x_test= x_train/255.0, x_test/255.0
   '''Budowanie modelu'''
   model=tf.keras.Sequential()
   model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # Splaszcza dane wejsciowe z 2 wym tablicy do 1 wym
   model.add(tf.keras.layers.Dense(128, activation='relu')) # Wymiar 128, activation to funkcja 
   #model.add(tf.keras.layers.Dropout(0.2)) #Losowa daje 0 do wejscia, ogranzicza overfitting
   model.add(tf.keras.layers.Dense(10))
    
   '''Trenowanie modelu'''
   predictions=model(x_train[:1]).numpy()
   loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

   model.compile(optimizer='Adam', loss=loss_fn, metrics=['accuracy'])   
   
   model.fit(x_train, y_train, epochs=1)
   
   '''Testowanie Wyszkolonego modelu'''
   model.evaluate(x_test, y_test, verbose=2)
   
   probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
   
   return probability_model

def Clasificate(model, image, img_to_print):
    #class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    predictions=model.predict(image)
    what_nr=numpy.argmax(predictions)
    plt.figure()
    plt.xlabel('Predicted image:' +  ' ' + class_names[what_nr] + '  ' + str(numpy.max(predictions)*100) + '%')
    plt.imshow(img_to_print)
    plt.show()
    
    
if __name__=="__main__":
    main()
