import cv2,os 
import numpy as np
import PIL.Image
import PIL.ImageTk
import pandas as pd
import datetime, time


face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'C:\Users\Indra\Downloads\Kp\Project\face_trainner\trainner.yml')
cap = cv2.VideoCapture(1) #menangkap object 
#membaca file mahasiswa.csv    
font=cv2.FONT_HERSHEY_SIMPLEX
df=pd.read_csv(r'C:\Users\Indra\Downloads\Kp\Project\Mahasiswa\mahasiswa.csv')
col_names = ['Id', 'Name', 'Date','Time']
attendance = pd.DataFrame(columns=col_names)

while(True):
    # Pengambilan bingkai(frame) 
    ret, frame = cap.read()
    # Operasi pada frame datang ke sini
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #rectangle (empat persegi panjang)
        color = (255, 0, 0) #BGR
        stroke = 2
        end_color_x = x + w
        end_color_y = y + h
        cv2.rectangle(frame, (x,y), (end_color_x,end_color_y), color, stroke)
        #print(x,y,w,h)
        #color camera
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #incrementing sample number
        #sampleNum = sampleNum+1
        Id, conf = recognizer.predict(roi_gray)
        if (conf < 50):
            time_s = time.time()
            date = str(datetime.datetime.fromtimestamp(time_s).strftime('%Y-%m-%d'))
            timeStamp = datetime.datetime.fromtimestamp(time_s).strftime('%H:%M:%S')
            nama = df.loc[df['Id'] == Id]['Name'].values
            name_get = str(Id) + "_" + nama
            attendance.loc[len(attendance)] = [Id, nama, date, timeStamp]
        else:
            Id = 'Unknown'
            name_get = str(Id)
        if (conf > 75):
            noOfFile = len(os.listdir(r"C:\Users\Indra\Downloads\Kp\Project\ImagesUnknow"))+1
            cv2.imwrite("ImagesUnknown\Image " + str(noOfFile) + ".jpg", frame[y:y+h,x:x+w])
            print("images Unknown")
        cv2.putText(frame,str(name_get),(x,y),font ,1 ,(255,255,255), 2, cv2.LINE_AA)
    attendance=attendance.drop_duplicates(subset=['Id'], keep='first')
    # Tampilkan frame yang dihasilkan 100 milisecond
    cv2.imshow('frame', frame)
    if (cv2.waitKey(1) == ord('q')):
        break
# Setelah semuanya selesai, simpan ke dalam file csv
time_s = time.time()
date = str(datetime.datetime.fromtimestamp(time_s).strftime('%Y-%m-%d'))
timeStamp = datetime.datetime.fromtimestamp(time_s).strftime('%H:%M:%S')
Hour, Minute, Second = timeStamp.split(":")
fileName = r"C:\Users\Indra\Downloads\Kp\Project\attendance\attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
#fileName = r"C:\Users\Indra\Downloads\Kp\Project\attendance\attendance_" + date + ".csv"
attendance.to_csv(fileName, index=False)

# attendance=attendance.drop_duplicates(keep='first',subset=['Id'])
# fileName="attendance/in.json"
# attendance.to_json(fileName,orient="index")

cap.release() #menutup kembali kamera
cv2.destroyAllWindows()
print("Attendance tracked") 
