from ftplib import error_temp
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button

fig, ax = plt.subplots()
fig.subplots_adjust( bottom=0.35)

lr = 0
errm = 0
epM = 0

t = np.arange(-1.0, 1.0, 0.001)
#l, = ax.plot(t, t)

def submit_lr(expression):
    #print(expression)
    lr = float(expression)

def submit_Errm(expression):
    #print(expression)
    errm = float(expression)

def submit_ep(expression):
    #print(expression)
    epM = float(expression)

def run():
    axbox = fig.add_axes([0.2, 0.2, 0.2, 0.075])
    text_box_Lr = TextBox(axbox, "Learning Rate","center")
    text_box_Lr.on_submit(submit_lr)
    text_box_Lr.set_val("0.1")  

    axbox_errm = fig.add_axes([0.2, 0.1, 0.2, 0.075])
    text_box_Errm = TextBox(axbox_errm, "Error minimo","center")
    text_box_Errm.on_submit(submit_Errm)
    text_box_Errm.set_val("0.1")  

    axbox_ep = fig.add_axes([0.2, 0.001, 0.2, 0.075])
    text_box_ep = TextBox(axbox_ep, "Epocas maximas","center")
    text_box_ep.on_submit(submit_ep)
    text_box_ep.set_val("100")  

    ax_iniW = fig.add_axes([0.7, 0.2, 0.2, 0.075])
    b_iniW = Button(ax_iniW, 'Inicializar W')
    
    ax_ada = fig.add_axes([0.7, 0.1, 0.2, 0.075])
    b_ada = Button(ax_ada, 'Adaline')

    ax_mat = fig.add_axes([0.7, 0.001, 0.2, 0.075])
    b_mat = Button(ax_mat, 'Matriz')
    
    plt.show()
    

if __name__ == '__main__':
    
    run()
