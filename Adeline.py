import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from Perceptron import Perceptron_
#import rhinoscriptsytnax as rs
from tkinter import *
from matplotlib.widgets import TextBox
from tabulate import tabulate


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
        self.point_calculado=[]
        self.graphic_points = np.arange(-10, 10, .1)
        self.epocas_totales_realizadas=0
    def get_graphic_points(self):
        return  self.graphic_points
    def get_pocas_totales_realizadas(self):
        return self.epocas_totales_realizadas
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
                line_func = -((self.W[1]/self.W[2]) * self.graphic_points) - (self.W[0]/self.W[2])
                self.point_calculado.append(line_func)
            errorMed = errorMed/self.m
            em+=1
        line_func = -((self.W[1]/self.W[2]) * self.graphic_points) - (self.W[0]/self.W[2])
        self.point_calculado.append(line_func)
        self.epocas_totales_realizadas=em
        if self.epochM==em:
            return self.point_calculado, -1 #murio
        else:
            return self.point_calculado, 1#jalo


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


    def matrizConv(self):
        clase_0 = 0
        clase_1 = 0
        falso_0 = 0
        falso_1 = 0
        for i in range(0,self.m):
            
            if self.Y[i] == 0 and self.y[i,0] < 0.5: #queria 0 obtuve 0
                    clase_0 += 1
            elif self.Y[i] == 1 and self.y[i,0] >= 0.5: #queria 1 obtuve 1
                    clase_1 += 1
            elif self.Y[i] == 0 and self.y[i,0] >= 0.5: #queria 0 obtuve 1
                    falso_1 += 1
            elif self.Y[i] == 1 and self.y[i,0] < 0.5: #queria 1 obtuve 0
                    falso_0 += 1

        datos = [['Datos: '+str(self.m), 'PRED 0', 'PRED 1',' '],
                ['REAL 0', str(clase_0), str(falso_1),str(clase_0+falso_1)],
                ['REAL 1', str(falso_0), str(clase_1),str(clase_1+falso_0)],
                [' ', str(clase_0+falso_0),str(clase_1+falso_1),' ']]
        print(tabulate(datos, tablefmt='fancy_grid'))

    def get_W(self):
        return self.W

    