import cv2
import json
import operator
import webbrowser
import numpy as np
from cv2 import data
from cv2.data import haarcascades
from keras.models import load_model
from youtubesearchpython import VideosSearch

#takes a photo
videoCaptureObject = cv2.VideoCapture(0)
result = True
while(result):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite("NewPicture.jpg",frame)
    result = False
videoCaptureObject.release()
cv2.destroyAllWindows()


image = cv2.imread('NewPicture.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faceCascade = cv2.CascadeClassifier(haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(20, 20),
    flags = cv2.CASCADE_SCALE_IMAGE)



xs= 0
ys= 0
width = 48
height = 48
# Draw a rectangle around the faces
for (x, y, w, h) in faces:

	xs= x
	ys= y
	width = w
	height = h

	break



cv2.waitKey(0)


crop_img = image[ys:ys+height, xs:xs+width]
cv2.waitKey(0)


resizedimg = cv2.resize(crop_img, (48,48))

cv2.waitKey(0)


gray48 = cv2.cvtColor(resizedimg, cv2.COLOR_BGR2GRAY)
cv2.imshow("face", gray48)
cv2.waitKey(0)


x_train = np.array([])

x_train = np.append(x_train,  gray48)

x_train = np.reshape(x_train, (1,48,48,1))

x_train = x_train.astype('float32')

x_train /= 255

model = load_model('models/CNNModelV3.h5')

prediction = model.predict(x_train)
print(prediction[0])



index, value = max(enumerate(prediction[0]), key=operator.itemgetter(1))

print(value)
print(index)






NNoutput = index

emotions = ("Angry", "Happy", "Sad", "Neutral")

#searches for video
videosSearch = VideosSearch(emotions[NNoutput] + ' pop music', limit = 1)
data2 = videosSearch.result()

#plays the video
webbrowser.open(data2['result'][0]['link'])