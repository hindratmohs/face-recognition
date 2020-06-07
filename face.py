import csv
import cv2
import os
import numpy as np
from PIL import Image
import pickle

#def takeimages():
name = input("Enter Your Name: ")
Id = input("Enter Your Id: ")
    

face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_alt2.xml')
#recognizer = cv2.face.LBPHFaceRecognizer_create()
cap = cv2.VideoCapture(0) #menangkap object 
sampleNum = 0


while(True):
    # Pengambilan bingkai(frame) demi bingkai
    ret, frame = cap.read()
    # Operasi pada frame datang ke sini
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #incrementing sample number
        sampleNum = sampleNum+1
        cv2.imwrite("images\ " + name +"."+Id + '.'+ str(sampleNum) + ".jpg", roi_gray) #menyimpan Gambar
        print("Berhasil menyimpan")

        #rectangle
        color = (255, 0, 0) #BGR
        stroke = 2
        end_color_x = x + w
        end_color_y = y + h
        cv2.rectangle(frame, (x,y), (end_color_x,end_color_y), color, stroke)
            
    # Tampilkan frame yang dihasilkan 100 milisecond
    cv2.imshow('frame',frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 100
    elif sampleNum > 60:
        break
# When everything done, release the capture
cap.release() #menutup kembali kamera
cv2.destroyAllWindows()
#save csv
row = [Id , name]
with open('Mahasiswa\mahasiswa.csv', 'a+') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)
csvFile.close()
res = "Images Saved for ID : " + str(Id) + " Name : " + name + " saved"
print(res)

