import numpy as np
import cv2
from firebase import firebase
import time
import datetime
import tensorflow as tf
import os



#load the model
path=os.getcwd()
path=path[0:len(path)-6]+"Training\\prediction.h5"
model_pred=tf.keras.models.load_model(path)

#connect to firebase
firebase=firebase.FirebaseApplication("https://attendance-system-3d51f.firebaseio.com/")

#get current date
x = datetime.datetime.now()
Date=x.strftime("%d")+'-'+x.strftime("%B")+'-'+x.strftime("%Y")
#print(Date)


#classify the faces and return an interger value
def classification(img,model):

    test=list()

    #resize the image and form appropriate shape for testing
    img=cv2.resize(img,(100,100))
    test.append(img)
    test=np.array(test).reshape(-1,100,100,3)
    test=test.astype('float32')/255
    prediction=model.predict_proba(test)
    inc=np.argmax(prediction)
    #print(f"{inc} : {prediction[0][inc]}")
    if prediction[0][inc]>=0.80:
        return inc
    else:
        return -1
 
#validation function to write into database(Firebase)
def validate(path,data):

	flag=0
	#print(data)

	#extract list of all students
	result=firebase.get(path,'')

	#print(result)


	if result==None:
		result=firebase.post(path,data)
		return

	else:

		for ids in result.keys():
			if result[ids]['name']==data['name']:
				if result[ids]['exit']==1:
					flag=1
					p=ids+'/exit'
					result=firebase.put(path,p,0)
					p=ids+'/entry'
					result=firebase.put(path,p,1)
					break

				if result[ids]['entry']==1:
					flag=1
					break

		if flag==0:
			result=firebase.post(path,data)
			return
	
#list of all the students.
students=["Student1","Student2","Student3","Student4","Student5"]

#haar casecade of facedetection.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#create objetc for capturing frames
cap = cv2.VideoCapture(0)

#start of with a frame
ret, prev = cap.read()


#run for infinite amout of time
while 1:

    #read a frame
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    result=-1
    name=""
    data=dict()

    #the entire thing gets triggered when there is change in background
    if (np.array(prev)-np.array(img)).tolist() !=0:

        #searches for face in the frame
        for (x,y,w,h) in faces:

            #cover it with a rectangle
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            #crop the face only
            roi_color = img[y:y+h, x:x+w]


            #classification as per whose face is that.
            result=classification(roi_color,model_pred)
            if result ==-1:
                pass

            else:
                #create a dictionary of credentials
                data['name']=students[result]
                data['entry']=1
                data['exit']=0
                data['time']=time.time()
                data['duration']=0

                #path to nosql database
                path='/attendance-system-3d51f/'+Date
                #print(data)

                #do the validation of the database
                validate(path,data)
                result=-1
                data=dict()
        

    #show the images
    cv2.imshow('img',img)

    prev=img

    #if esc is pressed the server gets dowm.
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#all cameras are released and windows are destroyed
cap.release()
cv2.destroyAllWindows()