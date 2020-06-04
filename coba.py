import os

folder = os.curdir('masukan nama : ')
newpath = r'C:\Users\Indra\Downloads\Kp\Project\images'
if not os.path.exists(folder):
    os.makedirs(newpath, folder)