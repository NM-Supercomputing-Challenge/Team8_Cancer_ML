from tkinter import *
import tkinter.font as tkFont

window = Tk()

titleFont = tkFont.Font(family="Georgia",size=50)

# Make title
title = Label(text="Data Page",
              font=titleFont,
              fg="#11447a")

title.place(relx=0.4,rely=0)

font = tkFont.Font(family="Georgia",size=20)

label = Label(text="Enter Data File",font=font)
label.place(relx=0.1,rely=0.1)

entryBox = Entry()
entryBox.place(relx=0.1,rely=0.15)

label2 = Label(text="Enter another Data File",font=font)
label2.place(relx=0.1, rely=0.2)

entryBox2 = Entry()
entryBox2.place(relx=0.1,rely=0.25)

label3 = Label(text="Enter a string or list with target variable(s)",font=font)
label3.place(relx=0.1,rely=0.3)

entryBox3 = Entry()
entryBox3.place(relx=0.1,rely=0.35)

window.mainloop()