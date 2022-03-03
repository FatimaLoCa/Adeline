import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from Perceptron import Perceptron_
#import rhinoscriptsytnax as rs
from tkinter import *
from matplotlib.widgets import TextBox

def sigmoid(z):
    return (1 / (1 + np.exp(-1 * z)))


class Adeline_:
    def __init__(self, W, matriz, theta, epochM, errorMin):
        self.W = W 
        self.theta = theta 
        self.epochM = epochM
        self.errorMin = errorMin

        self.matriz = matriz #Set de datos
        self.X = matriz[:, :2] # Separa la matriz 
        self.Y = matriz[:,2]
        
        (self.m,self.n) = self.X.shape # m - Filas, n - Columnas
        self.y = np.zeros((self.m,1)) # Vector para Y obtenidas, calculadas con sigmoid(pw)
        self.ones= np.ones((self.m,1)) 
        self.y_ob = []

    def iniciar(self):
        self.X = np.hstack((self.ones,self.X)) #agrega los 1
        self.n += 1
       
        
        error = 0
        errorMed = 2
        em = 0 #Contador de epocas
        
        while errorMed >= self.errorMin and em < self.epochM: #mientras no se ha terminado y no se cumplan las epocas
            errorMed = 0
            for i in range(0,self.m):
                pw_ = self.pw(self.X[i],self.W)
                have = sigmoid(pw_)
                self.y[i] = have
                error = self.Y[i] - have
                errorMed += error*error
                self.change_W(error, self.X[i], have)
            errorMed = errorMed/self.m
            em+=1

        #print(em)
        
        

        for j in range(0,self.m):
            self.y_ob.append(self.y[j,0]) # Esta Y es la que se usa para plotear, es una lista
        self.plotear()
    
    def change_W(self, error, x, y):
        nw = [0,0,0]
        for i in range(0,self.n):
            self.W[i] = self.W[i] + (self.theta * error * (y * (1 - y)) * x[i])
            
        

    def pw(self,x,w):
        pw_ = 0
        for i in range(0,self.n):
            pw_ += x[i] * w[i]

        return pw_

    
    def mostrar(self):
        print('W0 = %4f'% self.W[0]) 
        print('w1 = %4f'% self.W[1])
        print('W2 = %4f'% self.W[2])

    def plotear(self):
        dato1 = [ 1, 1, 0]
        dato2 = [-1, -1, 1]
        have = self.pw(dato1,self.W)
        have2 = self.pw(dato2,self.W)
        plt.plot(dato1[0],dato1[1] , 'ro')
        plt.plot(dato2[0],dato2[1] , 'bo')
        clase0 = []
        clase1 = []
        for i in range(0,self.m):
            if self.Y[i] == 0:
                clase0.append(self.X[i])
            else:
                clase1.append(self.X[i])

        v_clase0 = np.array(clase0)
        v_clase1 = np.array(clase1)

        plt.plot(v_clase0[:,1],v_clase0[:,2] , 'ro') # Clase 0 ROJO
        plt.plot(v_clase1[:,1],v_clase1[:,2] , 'bo') # Clase 1 AZUL

        for i in range(0,self.m):
            color = round(self.y_ob[i], 2)
            
            if(self.y[i] >= 0.5): #Mayor a 0.5 - Clase 1
                plt.plot(self.X[i][1],self.X[i][2], marker = "*", color = (0, color, 0))
            else: # Clase 0                                                R   G   B
                plt.plot(self.X[i][1],self.X[i][2], marker = "*", color = (1-color, 0, 0.2))
        
        graphic_points = np.arange(-10, 10, .1)
        line_func = -((self.W[1]/self.W[2]) * graphic_points) - (self.W[0]/self.W[2])
        plt.plot(graphic_points, line_func)
        
        plt.axis([-1, 1, -1, 1])
        plt.show()
            
    
def run():
    matriz = np.loadtxt('dataset_Perceptron.txt',delimiter = ',')
    em = 500
    theta = 0.4
    W = [random.random() for i in range(1,4)]
    err = 0.02

    #p = Perceptron_(W,matriz,theta,em)
    #p.iniciar()

    a = Adeline_(W,matriz,theta,em,err)
    a.iniciar()

if __name__ == '__main__':
    run()


