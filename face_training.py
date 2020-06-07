import cv2
import os
import numpy as np
from PIL import Image
import pickle



face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier(r'C:\Users\Indra\Downloads\Kp\Project\cascades\data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() #algoritma recognizer
def getImagesWithLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empty face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L') #Loading the images in Training images and converting it to gray scale
        imageNp=np.array(pilImage,'uint8') #Now we are converting the PIL image into numpy array
        Id=int(os.path.split(imagePath)[-1].split(".")[1]) #getting the Id from the image
        faces=face_cascade.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples, Ids
#simpan data sampel
faces, Ids = getImagesWithLabels('images')
recognizer.train(faces, np.array(Ids))
recognizer.save(r'C:\Users\Indra\Downloads\Kp\Project\face_trainner\trainner.yml')
print("training image succsess")
# current_id = 0
# label_ids = {}
# y_labels = []
# x_train = []

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# image_dir = os.path.join(BASE_DIR, "images")

# for root, dirs, files in os.walk(image_dir):
#     for file in files:
#         if file.endswith("png") or file.endswith("jpg"):
#             path = os.path.join(root, file)
#             label = os.path.basename(root).replace(" ", "-").lower()
#             #print(label, path)
#             if not label in label_ids:
#                 label_ids[label] = current_id
#                 current_id += 1
#             id_ = label_ids[label]
#             #print(label_ids)
#             #y_labels.append(label) # some number
#             #x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
#             pil_image = Image.open(path).convert("L") # grayscale
#             size = (550, 550)
#             final_image = pil_image.resize(size, Image.ANTIALIAS)
#             image_array = np.array(final_image, "uint8")
#             #print(image_array)
#             faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

#             for (x,y,w,h) in faces:
#                 roi = image_array[y:y+h, x:x+w]
#                 x_train.append(roi)
#                 y_labels.append(id_)


#print(y_labels)
#print(x_train)

# with open(r"C:\Users\Indra\Downloads\Kp\Project\pickles\face-labels.pickle", 'wb') as f:
# 	pickle.dump(label_ids, f)

# #recognizer.train(x_train, np.array(y_labels))
# recognizer.save(r"C:\Users\Indra\Downloads\Kp\Project\face_trainner\trainner.yml")