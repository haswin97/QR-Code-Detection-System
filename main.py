from tkinter import *
from tkinter.filedialog import askopenfilename

import tkinter.messagebox as msg #library utk menampilkan pesan pop-up

import globalvar #import coding globalvar
import detection #import coding detection


def button_open_exec():
    # try:
        globalvar.label_status.configure(text="Status : Ready...", font='Arial 9 bold')
        open_path = askopenfilename(defaultextension=".mov", title='Open a Video file',
                                   filetypes=(("MOV files", ".mov"), ("MPG files", ".mpg"), ("All files", "*.*")))

        if not string_not_blank(open_path):
            msg.showwarning('Warning', 'Open a Video file Failed !')
            return

        detected = detection.process_detection(open_path)
        if detected:
            msg.showinfo('Result of QR-Code Detection', globalvar.result_qrcode)
        else:
            msg.showwarning('Result of QR-Code Detection', 'NO QR-Code DETECTED !!')


        globalvar.label_status.configure(text="Status : Ready ...", font='Arial 9 bold')
    # except:
    #     msg.showerror("Error", "Open a Video File Error")
    #     globalvar.label_status.configure(text="Status : Ready...", font='Arial 9 bold')


def string_not_blank(string_value):
    "if string is empty or none return false"
    return bool(string_value and string_value.strip()) #utk keadaan string null atau empty

def main():
    # initialize Tkinter
    globalvar.root = Tk()
    globalvar.root.resizable(0, 0)
    globalvar.root.title('QR-Code Detection by Ruben, Telkom University - 2020')

    globalvar.label_status = Label(globalvar.root, text="Status : Ready ...", font='Arial 9 bold')
    globalvar.label_status.place(x=10, y=520)

    globalvar.button_open = Button(globalvar.root,text="Open a Video File", command=lambda: button_open_exec(),
                         state=NORMAL, height=1, width=15)
    globalvar.button_open.place(x=10, y=550)

    label_title = Label(globalvar.root,text="QR-Code Detection",font='Arial 12 bold')
    label_title.place(x=348,y=10)


    width = 860     #utk ubah ukuran GUI (line 56 - 60)
    height = 590

    X = int(globalvar.root.winfo_screenwidth() / 2 - width / 2)
    Y = int(globalvar.root.winfo_screenheight() / 2 - height / 2)

    sizeTk = str(width) + 'x' + str(height)
    globalvar.root.geometry('{}+{}+{}'.format(sizeTk, X, Y))

    globalvar.root.update()
    globalvar.root.mainloop()


# ======= MAIN FUNCTION
if __name__ == '__main__':
    main()
