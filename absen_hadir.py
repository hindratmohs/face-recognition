import tkinter as tk
import cv2, os
import csv
import numpy as np
from PIL import Image,ImageTk
import pandas as pd
from tkinter import messagebox
import PIL.ImageTk
import datetime, time

#tampinlan window 
window = tk.Tk()
window.geometry("400x400")
window.title("Absensi kehadiran")

#frame
# frame1 = tk.Frame(master=window, width=50, height=50, bg="gray")
# frame1.pack()


#judul
message = tk.Label(window, text="ABSENSI" ,fg="black"  ,width=15  ,height=1,font=('Calibri',20))
message.place(x=100, y=20)

#input nama  
L1 = tk.Label(window, text = "notifikasi  ")
L1.place(x=50, y=100)
E1 = tk.Label(window, text="",width=30,bg="white",fg="black")
E1.place(x=150, y=100)

L2 = tk.Label(window, text = "kehadiran   ")
L2.place(x=50, y=230)
E2 = tk.Label(window, text="",width=35,bg="white",fg="black")
E2.place(x=130, y=230)



#fungsi text
def clear1():
    E1.delete(0, 'end')    
    res = ""
    E1.configure(text= res)

def clear2():
    E2.delete(0, 'end')    
    res = ""
    E2.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

#fungsi face recognaze
face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_default.xml')
def takeimages():
    name = (E1.get())
    Id = (E2.get())
    if not Id:
        # res= "Please masukkan Id"
        # message.configure(text=res)
        MsgBox=tk.messagebox.askquestion("Warning","Tolong memasukkan ID anda ",icon='warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo('Warning','Terimakasih...!')
    elif not name:
        # res="Please masukkan Nama"
        # message.configure(text = res)
        MsgBox = tk.messagebox.askquestion ("Warning","Tolong memasukkan Nama anda ",icon = 'warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo('Warning','Terimakasih...!')

    elif(is_number(Id) and name.isalpha()):
        df=pd.read_csv('Mahasiswa\mahasiswa.csv')
        if(df['Id'].astype(str).str.contains(str(Id)).any()==True):
            tk.messagebox.showinfo('warning',"User telah digunakan")
        else:
            face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_default.xml')
            #face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_alt2.xml')
            #recognizer = cv2.face.LBPHFaceRecognizer_create()
            cap = cv2.VideoCapture(1) #menangkap object 0 camera internal 1 camera external
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
                    #menambah jumlah sampel
                    sampleNum = sampleNum+1
                    #simpan data gambar muka ke dalam folder images
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
            res = "Images Saved for ID : " + str(Id) + " Name : " + name + " saved"
            row = [Id, name]
            with open('Mahasiswa\mahasiswa.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            # message.configure(text= res)
            print(res)
    # else:
    #     if(is_number(Id)):
    #         res = "Masukan Alphabetical Nama"
    #         message.configure(text= res)
    #     if(name.isalpha()):
    #         res = "Masukan Numeric Id"
    #         message.configure(text= res)

def TrainImages():
    #face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create() #algoritma recognizer
    #simpan data sampel
    faces, Ids = getImagesWithLabels('images')
    recognizer.train(faces, np.array(Ids))
    recognizer.save(r'C:\Users\Indra\Downloads\Kp\Project\face_trainner\trainner.yml')
    print("training images succsess")
    # res = "Save Success"
    clear1();
    clear2();
    # message.configure(text= res)
    tk.messagebox.showinfo('Completed','Save successfully!!')

def getImagesWithLabels(path):
    # face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_default.xml')
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #buat daftar wajah kosong
    faceSamples=[]
    #buat daftar Id kosong
    Ids=[]
    #perulangan melalui semua jalur gambar dan memuat Id dan gambar
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L') #Memuat gambar dalam gambar Pelatihan dan mengubahnya menjadi skala abu-abu
        imageNp=np.array(pilImage,'uint8') #mengubah gambar PIL menjadi array numpy
        Id=int(os.path.split(imagePath)[-1].split(".")[1]) #mendapatkan Id dari gambar
        faces=face_cascade.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples, Ids

def siswa():
    root = tk.Tk()
    root.geometry()
    root.title("Daftar siswa")
    # tentukan lokasi file, nama file, dan inisialisasi csv

    f = open('Mahasiswa\mahasiswa.csv', 'r')
    reader = csv.reader(f)
    r = 0 
    # membaca baris per baris
    for col in reader:
        # MsgBox=tk.messagebox.askquestion('nama siswa', row)
        # print (row)
        c = 0
        for row in col:
            # i've added some styling
            label = tk.Label(root, width=15, height=1, fg="black", font=('times', 12),
                                bg="white", text=row, relief=tk.RIDGE)
            label.grid(row=r, column=c)
            c += 1
        r += 1
    root.mainloop()
    # tk.messagebox.askquestion("Warning",siswa)
    # # tentukan lokasi file, nama file, dan inisialisasi csv
    # f = open('Mahasiswa\mahasiswa.csv', 'r')
    # reader = csv.reader(f)
    # r = 0 
    # # membaca baris per baris
    # for col in reader:
    #     # MsgBox=tk.messagebox.askquestion('nama siswa', row)
    #     # print (row)
    #     c = 0
    #     for row in col:
    #         # i've added some styling
    #         label = tk.Label(window, width=7, height=1, fg="black", font=('times', 12),
    #                              bg="white", text=row, relief=tk.RIDGE)
    #         label.grid(row=r, column=c)
    #         c += 1
    #     r += 1
    
    # menutup file csv
    #f.close()

def quit():
    global window
    window.quit()

def user():
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
    # tk.messagebox.showinfo('kehadiran','Kehadiran Masuk,Terimakasih ')
    # attendance=attendance.drop_duplicates(keep='first',subset=['Id'])
    # fileName="attendance/in.json"
    # attendance.to_json(fileName,orient="index")

    cap.release() #menutup kembali kamera
    cv2.destroyAllWindows()
    res=attendance
    E2.configure(text=res)
    res = "Masuk"
    E1.configure(text=res)
    tk.messagebox.showinfo('kehadiran','Terimakasih anda Telah Hadir')
    print("Attendance tracked") 

def camera():
    cap = cv2.VideoCapture(1)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# def gambar():
#     img = ImageTk.PhotoImage(Image.open("images/billgets.1293108.1.jpg"))
#     img_Label = tk.Label(image=img)
#     img_Label.pack()

#Tombol
# B = tk.Button(window, text = "Save ", command = TrainImages)
# B.place(x = 185,y = 350)

B2 = tk.Button(window, text = "Hadir", command = user)
B2.place(x = 170,y = 170)

# Menu Bar
menu = tk.Menu(window)
window.config(menu=menu)

subMenu=tk.Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=subMenu)
# subMenu.add_command(label="Daftar Mahasiswa", command = siswa)
# subMenu.add_command(label="Daftar Hadir")
# subMenu.add_command(label="Daftar Pulang")
subMenu.add_command(label="Cek camera", command = camera)
subMenu.add_command(label="Exit", command = quit)


window.mainloop()