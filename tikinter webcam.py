import tkinter
import PIL.Image, PIL.ImageTk
import numpy as np
import argparse
import cv2
import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        
        self.emotion_dcit = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        self.radio_value = tkinter.IntVar()
        self.radio_value.set(0)
        self.faces = [("Trump", 0), ("RonaldMcDonald", 1), ("NicholasCage", 2), ("Emoji", 3), ("Shrek", 4), ("Pepe", 5)]
        self.selected_face = 4

        self.resized = ""

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        
        
        for index, face in enumerate(self.faces):
            tkinter.Radiobutton(window, text=face[0], width=50, variable = self.radio_value, command=self.radio_choice, value=index).pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

 
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            if emotion_dict[maxindex] == "Angry":
                im = cv2.imread('faces/' + self.faces[self.radio_value.get()][0] + '/angry.png')
            elif emotion_dict[maxindex] == "Disgusted":
                im = cv2.imread('faces/' + self.faces[self.radio_value.get()][0] + '/disgusted.png')
            elif emotion_dict[maxindex] == "Fearful":
                im = cv2.imread('faces/' + self.faces[self.radio_value.get()][0] + '/fearful.png')
            elif emotion_dict[maxindex] == "Happy":
                im = cv2.imread('faces/' + self.faces[self.radio_value.get()][0] + '/happy.png')
            elif emotion_dict[maxindex] == "Neutral":
                im = cv2.imread('faces/' + self.faces[self.radio_value.get()][0] + '/neutral.png')
            elif emotion_dict[maxindex] == "Sad":
                im = cv2.imread('faces/' + self.faces[self.radio_value.get()][0] + '/sad.png')
            elif emotion_dict[maxindex] == "Surprised":
                im = cv2.imread('faces/' + self.faces[self.radio_value.get()][0] + '/suprised.png')
            else:
                im = cv2.imread('faces/' + self.faces[self.radio_value.get()][0] + '/neutral.jpg')

            self.resized = cv2.resize(im, (h, w))

            frame[y: y+h, x: x+w] = self.resized.astype(object)
            
 
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)


        self.window.after(self.delay, self.update)

    def radio_choice(self):
        pass
 
 
class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
 
# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
