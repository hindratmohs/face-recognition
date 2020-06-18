import tkinter as tk
import csv
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
        label = tk.Label(root, width=7, height=1, fg="black", font=('times', 12),
                            bg="white", text=row, relief=tk.RIDGE)
        label.grid(row=r, column=c)
        c += 1
    r += 1

# myLabel = tk.Label(root, siswa)
# myLabel.pack()


root.mainloop()