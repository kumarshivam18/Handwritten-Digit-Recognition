import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
mnist= tf.keras.datasets.mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data() 
'''

'''
x_train = tf.keras.utils.normalize(x_train , axis = 1)
x_test = tf.keras.utils.normalize(x_test , axis = 1)
'''

'''
model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train, epochs=3)

model.save('C:\\Users\\SHIVAM\\Desktop\\Mini project 4th semester 2023\\handwritten.model')
'''
model= tf.keras.models.load_model('C:\\Users\\SHIVAM\\Desktop\\Mini project 4th semester 2023\\handwritten.model')

'''
loss , accuracy  =model.evaluate(x_test,y_test)
print(accuracy)
print(loss)
'''

image_number = 1
image_directory = r"C:\Users\SHIVAM\Desktop\Mini project 4th semester 2023\Digits"

while os.path.isfile(os.path.join(image_directory, f"input{image_number}.png")):
    try:
        img = cv2.imread(os.path.join(image_directory, f"input{image_number}.png"))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("This Digit is", np.argmax(prediction))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1

