from tkinter import *
import tkinter.font as tkFont
from tkinter import ttk

with open("project_variables.txt","r") as projectVars:
    vars=projectVars.readlines()

window = Tk()

main_frame = Frame(window)
main_frame.pack(fill=BOTH,expand=1)

canvas = Canvas(main_frame)
canvas.pack(side=LEFT, fill=BOTH, expand=1)

# Add scrollbars to the canvas
scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

scrollbar_x = ttk.Scrollbar(main_frame,orient=HORIZONTAL, command=canvas.xview)
scrollbar_x.pack(side=BOTTOM, fill=X)

# Configure the canvas
canvas.configure(xscrollcommand=scrollbar_x.set)
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

second_frame = Frame(canvas)
canvas.create_window((0,0), window=second_frame, anchor="nw")

# get screen dimensions
screenHeight = second_frame.winfo_screenheight()
screenWidth = second_frame.winfo_screenwidth()

# make font for title
fontTitle = tkFont.Font(family="Georgia",size=20)

# make font for variable labels
varFont = tkFont.Font(family="Berlin Sans FB",size=12)

# make title
title = Label(second_frame,text="Data Page",font=fontTitle)
title.grid(padx=screenWidth/2,pady=20)

def makeButton(name):
    label = Label(second_frame,text=name,font=varFont,fg="#6e7372")
    label.grid(pady=40)

    button = Button(second_frame,text="True",width=20,height=2)
    button.grid()

    button = Button(second_frame,text="False",width=20,height=2)
    button.grid()

def makeEntry(name):
    label = Label(second_frame,text=name,font=varFont,fg="#6e7372")
    label.grid(pady=40)

    entry = Entry(second_frame)
    entry.grid()

for var in vars:
    var = str(var)
    # checking if variable is a string
    i = var.find('"')

    # check if string contains tuple
    t = var.find("(")
    t2 = var.find(")")

    # check if string contains numbers
    n = any(char.isdigit() for char in var)

    # Only use part of variable before equal sign
    var = var.rsplit('=', 1)[0]

    if i == -1 and t == -1 and t2 == -1 and n == False:
        makeButton(var[:-1])
    else:
        makeEntry(var[:-1])

window.mainloop()
