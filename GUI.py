from tkinter import *
import tkinter.font as tkFont
from tkinter import ttk

with open("project_variables.txt","r") as projectVars:
    vars=projectVars.readlines()

print(vars)

window = Tk()

main_frame = Frame(window)
main_frame.pack(fill=BOTH,expand=1)

canvas = Canvas(main_frame)
canvas.pack(side=LEFT, fill=BOTH, expand=1)

# Add a scrollbars to the canvas
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

def makeButton(name):
    label = Label(second_frame,text=name)
    label.pack()

    button = Button(second_frame,text="True")
    button.pack()

    button = Button(second_frame,text="False")
    button.pack()

def makeEntry(name):
    label = Label(second_frame,text=name)
    label.pack()

    entry = Entry(second_frame)
    entry.pack()

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


