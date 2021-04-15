# identifying the faces 
import cv2 
import sys
from numpy import *
import os
import time

haar_file='haarcascade_frontalface_default.xml'
datasets='datasets'

# Part 1: Create fisherRecognizer 
print('\nRecognizing Face Please Be in sufficient Lights...\n') 

# Create a list of images and a list of corresponding names and create the id's associated with each label
(images,lables,names,id)=([],[],{},0) 
for (subdirs,dirs,files) in os.walk(datasets): 
	for subdir in dirs: 
		names[id]=subdir 
		#image directory
		subjectpath=os.path.join(datasets,subdir) 
		for filename in os.listdir(subjectpath): 
			path=subjectpath+'/'+filename 
			#names of the images
			lable=id
			images.append(cv2.imread(path,0)) #verify this image and turn it into numpy array
			lables.append(int(lable)) 
		id+=1
(width,height)=(130,100) 

# Create a Numpy array from the two lists above to get the numbers from the images
(images,lables) = [array(lis) for lis in [images,lables]] 

# OpenCV trains a recognizer from the images 
# NOTE FOR OpenCV2: remove '.face' 
recognizer=cv2.face.LBPHFaceRecognizer_create() 
recognizer.train(images,lables) 

# Part 2: Use fisherRecognizer on camera stream 
face_cascade=cv2.CascadeClassifier(haar_file) 
webcam=cv2.VideoCapture(0,cv2.CAP_DSHOW) 
ispredicted=0
while True: 
	(_,image) = webcam.read() 
	gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
	faces=face_cascade.detectMultiScale(gray, 1.3, 5) 
	for (x,y,w,h) in faces: 
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2) 
		face=gray[y:y+h,x:x+w] 
		face_resize=cv2.resize(face,(width,height)) 
		# Try to recognize the face - flag ,confidence
		prediction=recognizer.predict(face_resize) 
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3) 
		#confidence
		if prediction[1]<100: 
			#frame,name and confidence,coordinates,font,1,font
			cv2.putText(image,'% s - %.0f' %(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0)) 
			ispredicted=1
			print("Hi",names[prediction[0]])
			print("You Can Access The Van\n")
		else: 
			cv2.putText(image,'Not Recognized',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0)) 

	cv2.imshow('OpenCV', image) 
	
	key=cv2.waitKey(10) 
	if key==27 or ispredicted==1: #if escape is pressed or face is recognised
		cv2.destroyAllWindows()
		webcam.release()
		break

if(ispredicted==1):
	print("Source: 0\n")
	print("Number Of Locations to Visit:")
	n=int(input())
	# visited array with all elements as zero
	vis=zeros(n+1,int)
	ans=99999999
	temp=[]
	final=[]
	tans=0
	# adj matrix
	dist=([])

	def dfs(pos):
		global vis,ans,tans
		global temp,final
		vis[pos]=1
		temp.append(pos)
		f=True
		for city in range(n+1):
			if(vis[city]==0):
				f=False
				tans=tans+dist[pos][city]
				dfs(city)
				tans=tans-dist[pos][city]
		if(f):
			temp.append(0)
			if(ans>tans+dist[pos][0]):
				final=temp.copy()
				ans=tans+dist[pos][0]
			temp.pop()
		temp.pop()
		vis[pos]=0

	l=zeros(n+1,int)
	for i in range(n+1):
		dist.append(l.copy())
	for i in range(1,n+1):
		for j in range(i):
			print("Dist Of",j,"From",i,"is:")
			distance=int(input())
			dist[i][j]=distance
			dist[j][i]=distance
	dfs(0)
	print("**********************************")
	print("Distance to Travel ->",ans,"km")
	print("**********************************")
	print("Path to Be Used:")
	x=len(final)
	for i in range(x-1):
		print("GO FROM",final[i],"TO",final[i+1])
	print("**********************************")
	
else:
	print("Sorry Your Face Couldn't Be Recognized")