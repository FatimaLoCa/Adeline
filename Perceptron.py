import random
from tkinter import Checkbutton
#from types import NoneType 
from matplotlib.widgets import Button
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from entry import Entry
import tkinter as tk


data=[]
fig, ax = plt.subplots()
ax_clean=ax

#plotea de -1 a 1
plt.xlim([0,4])
plt.ylim([0,4])
def paint():
    for entry in data:
        if entry.get_Clase() ==1:
            ax.scatter(x=entry.get_X(),y=entry.get_Y(),color='tab:cyan')
            #print("x posicion click",entry.get_X()," y pos clik",entry.get_Y())
        else:
            ax.scatter(x=entry.get_X(),y=entry.get_Y(),color='tab:pink')

class Index:

    def start(self, event):
        #matriz = np.loadtxt('dataset_Perceptron.txt',delimiter = ',')
        #print(matriz)
        
        matriz=[]
        entry=[]
        for obj in data:
            entry=[obj.get_X(),obj.get_Y(),obj.get_Clase()]
            matriz.append(entry)

        
        matriz=np.transpose(matriz)
        matriz=np.transpose(matriz)
        paint()
        #print(matriz)
        em = 100
        theta = .4
        W = [random.random() for i in range(1,4)]
        print(W)
        err = 0

        p = Perceptron_(W,matriz,theta,em)
        p.iniciar()
        #print("y_ob",p.y_ob)

    def mapear(self, event):
        print("dio click mapear")


class Perceptron_:
    def _init_(self, W, matriz, theta, epochM):
        self.W = W 
        self.matriz = matriz #Set de datos
        self.theta = theta 
        self.X = matriz[:, :2]
        self.Y = matriz[:,2]
        (self.m,self.n) = self.X.shape # m - Filas, n - Columnas
        self.y = np.zeros((self.m,1))
        self.epochM = epochM
        self.ones= np.ones((self.m,1))  *-1
        self.y_ob = []

    def iniciar(self):
        self.X = np.hstack((self.ones,self.X)) #agrega los -1
        self.n += 1
        done = False
        error=0
        em=0
        print("con -1",self.X)

        while done == False and em<self.epochM: #mientras no se ha terminado y no se cumplan las epocas
            done = True
            for i in range(0,self.m): 
                have = self.pw(self.X[i],self.W)
                error = self.Y[i] - have
                #print("have",have," deseada ",self.Y[i],"error",error)
                if error != 0:
                    done = False
                    self.change_W(error, self.X[i])
                    m=-(self.W[1]/self.W[2])
                    b=(self.W[0]/self.W[2])
                    calculados=[x2*m +b for x2 in self.X]
                    #line=ax.plot([self.X[0][1],self.Y[0]],[m*self.X[0][1]+b, m*self.Y[1]+b],color='tab:orange')
                    
                    line=ax.plot(self.X,calculados)
                    plt.pause(.1)
                    for i in line:
                        line_=line.pop(0)
                        line_.remove()
                    line_=line.pop(0)
                    line_.remove()
                    
            em += 1
            print("epocas",em, "W",self.W)
            if em == self.epochM:
                tk.messagebox.showerror(title="PERCEPTRON FALLO", message="No logramos converger 😢😢😢")
                plt.close()
                return
           
            
            
        m=-(self.W[1]/self.W[2])
        b=(self.W[0]/self.W[2])
        calculados=[x2*m +b for x2 in self.X]
                    #line=ax.plot([self.X[0][1],self.Y[0]],[m*self.X[0][1]+b, m*self.Y[1]+b],color='tab:orange')
        ax.plot(self.X,calculados,color='tab:red')        
        plt.draw()
        self.calcular_Y_ob()

    def calcular_Y_ob(self):
        for i in range(0,self.m):
            have = self.pw(self.X[i],self.W)
            self.y_ob.append(have)

    def change_W(self, error, x):
        nw = [0,0,0]
        for i in range(0,self.n):
            nw[i] = self.W[i] + (self.theta * error * x[i])
            #print("w i:",i,self.W[i],"theta",self.theta,"error ",error,"x[i]",x[i],"nuevo W",nw[i])
        self.W = nw

        

    def pw(self,x,w):
        pw_ = 0
        for i in range(0,self.n):
            pw_ += w[i] * x[i]

        if pw_ >= 0:
            return 1
        return 0
			
    def mostrar(self):
        print('W0 = %4f'% self.W[0]) 
        print('w1 = %4f'% self.W[1])
        print('W2 = %4f'% self.W[2])




def on_click(event):
    #print(event)
    x, y = event.x, event.y
    if(x<580 or y<420):
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
        
    else:
        pass
        
def window():
    callback = Index()
    axmapeo = plt.axes([0.90, 0.80, 0.1, 0.075])
    axstar = plt.axes([0.90, 0.90, 0.1, 0.075]) # pos_x, pos_y,

    button_start = Button(axstar, 'Start')
    button_map = Button(axmapeo, 'Mapeo')

    button_start.on_clicked(callback.start) 
    button_map.on_clicked(callback.mapear) 
    plt.connect('button_press_event', on_click)
    plt.show()


def run():
    pass


if __name__ == '__main__':
    window()