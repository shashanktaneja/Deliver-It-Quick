# Creating database, captures images and stores them in datasets folder under the folder name of sub_data 
import cv2
import sys
import numpy
import os 
haar_file='haarcascade_frontalface_default.xml'

# All the faces data will be present this folder 
datasets='datasets'

# Sub data sets of folder, for my faces I've used my name you can change the label here 
sub_data='shashank taneja'	

path=os.path.join(datasets,sub_data) 
if not os.path.isdir(path): 
	os.makedirs(path) 

# defining the size of images 
(width,height)=(130,100)	 

#'0' is used for my webcam, if you've any other camera attached use '1' like this 
face_cascade=cv2.CascadeClassifier(haar_file) #classifier contains the features to detect the faces 
webcam=cv2.VideoCapture(0,cv2.CAP_DSHOW) 

# The program loops until it has 30 images of the face. 
count=1
while count<30: 
	# _=flag to tell if the frame was detected correctly or not and image is the frame
	(_, image)=webcam.read()       
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	#detection of coordinates of face 
	faces=face_cascade.detectMultiScale(gray,1.3,4) 
	#creating a rectangle at location of face
	for(x,y,w,h) in faces: 
		#image,coordinates starting and ending,color of rectangle, width of rectangle
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		# region of interest of grey picture
		face=gray[y:y+h,x:x+w] 
		face_resize=cv2.resize(face,(width,height)) 
		cv2.imwrite('% s/% s.png' % (path,count),face_resize) 
	count+=1
	
	cv2.imshow('OpenCV',image) 
	key=cv2.waitKey(10) 
	if key == 27: 
		break
