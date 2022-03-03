
from pdb import line_prefix
from tkinter import Checkbutton
from types import NoneType
from matplotlib.widgets import Button
import numpy as np
from matplotlib.backend_bases import MouseButton
from entry import Entry
from Perceptron import Perceptron_
from Adeline import Adeline_
import random
import tkinter as tk
from ftplib import error_temp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button


fig, ax = plt.subplots()
data=[]
fig.subplots_adjust( bottom=0.35)
plt.xlim([-2,2])
plt.ylim([-2,2])
lr = 0.1
errm = 0.1
epM = 100
graphic_points = np.arange(-1, 1, .1)
t = np.arange(-1.0, 1.0, 0.001)

class aux:
        
    W=[0,0,0]
    line_w=""
    flag_line_W=False
    line_=""
    
aux_=aux()
def matriz(self,event):
    print("matriz")


def adaline_inicialize(event):
    if len(set_datos)==0:
        tk.messagebox.showerror(title="VERIFIQUE", message="NO TENEMOS SET DE DATOS 😢😢😢")
        return

    if aux.W[0]==0:
        aux.W= [random.random() for i in range(1,4)]
    adaline=Adeline_(aux.W,set_datos,lr,epM,errm)
        
    graphic_points,convergio=adaline.iniciar()
    points=adaline.get_graphic_points()
    for point in graphic_points:
        line=ax.plot(points,point)
        plt.pause(.1)
        for i in line:
            line_=line.pop(0)
            line_.remove()
    if convergio ==-1:
        tk.messagebox.showerror(title="Adaline FALLO", message="No logramos converger 😢😢😢")
        plt.close()
        return
    tk.messagebox.showinfo(title="Adaline CONVERGIO", message="logramos converger en:"+adaline.get_epocas_totales_realizadas()+" epocas 😍😍")



def init_w(event):
    if aux.flag_line_W:
        aux.line_=aux.line_w.pop(0)
        aux.line_.remove()
    aux.W=[random.random() for i in range(1,4)]
    line_func = -((aux.W[1]/aux.W[2]) * graphic_points) - (aux.W[0]/aux.W[2])
    aux.line_w=ax.plot(graphic_points,line_func)
    aux.flag_line_W=True


def entry_to_matriz():
    
        matriz=[]
        entry=[]
        for obj in data:
            entry=[obj.get_X(),obj.get_Y(),obj.get_Clase()]
            matriz.append(entry)

        
        matriz=np.transpose(matriz)
        matriz=np.transpose(matriz)
        return matriz

def paint():
    for entry in data:
        if entry.get_Clase() ==1:
            ax.scatter(x=entry.get_X(),y=entry.get_Y(),color='tab:cyan')
            #print("x posicion click",entry.get_X()," y pos clik",entry.get_Y())
        else:
            ax.scatter(x=entry.get_X(),y=entry.get_Y(),color='tab:pink')

def onclick(event):
    if event.inaxes != ax:
        return
    if event.button is MouseButton.LEFT:
        if event.xdata ==None or event.ydata ==None:
            return
        data.append(Entry(event.xdata,event.ydata,1))
        ax.scatter(x=event.xdata,y=event.ydata,color='tab:cyan')
        plt.draw()
    elif event.button is MouseButton.RIGHT:
        if event.xdata ==None or event.ydata ==None:
            return
        data.append(Entry(event.xdata,event.ydata,0))
        ax.scatter(x=event.xdata,y=event.ydata,color='tab:pink')
        plt.draw()
    

def submit_lr(expression):
    #print(expression)
    lr = float(expression)

def submit_Errm(expression):
    #print(expression)
    errm = float(expression)

def submit_ep(expression):
    #print(expression)
    epM = float(expression)


def window():
    
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
    b_iniW.on_clicked(init_w)

    ax_ada = fig.add_axes([0.7, 0.1, 0.2, 0.075])
    b_ada = Button(ax_ada, 'Adaline')
    b_ada.on_clicked(adaline_inicialize)

    ax_mat = fig.add_axes([0.7, 0.001, 0.2, 0.075])
    b_mat = Button(ax_mat, 'Matriz')
    b_mat.on_clicked(matriz)
    
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

if __name__ == '__main__':
    window()





