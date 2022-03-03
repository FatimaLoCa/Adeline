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

        

    def iniciar(self):
        self.X = np.hstack((self.ones,self.X)) #agrega los 1
        self.n += 1
       
        
        error = 0
        errorMed = 2
        em = 0 #Contador de epocas
        fig, ax = plt.subplots()
        plt.figure(1)
        self.plotear()
        while errorMed > self.errorMin and em < self.epochM: #mientras no se ha terminado y no se cumplan las epocas
            errorMed = 0
            for i in range(0,self.m):
                pw_ = self.pw(self.X[i],self.W)
                have = sigmoid(pw_)
                self.y[i] = have
                error = self.Y[i] - have
                
                errorMed += np.abs(error)
                
                
                self.change_W(error, self.X[i], pw_)

                m=-(self.W[1]/self.W[2])
                b=(self.W[0]/self.W[2])
                line=plt.plot([self.X[0][0],self.Y[0]],[m*self.X[0][1]+b, m*self.Y[1]+b],color='tab:orange')
                plt.pause(.1)
                line_del = line.pop(0)
                line_del.remove()

            errorMed = errorMed/self.m
            errores.append(errorMed)
            epocas.append(em)
            plt.figure(1)
            plt.plot(epocas,errores)
            plt.draw()
            em+=1
            
            #print(errorMed)
            em += 1
        m=-(self.W[1]/self.W[2])
        b=(self.W[0]/self.W[2])
        plt.plot([self.X[0][0],self.Y[0]],[m*self.X[0][1]+b, m*self.Y[1]+b],color='tab:orange')


        print(self.y)
        
        print(self.Y)
        print(em)

        # for j in range(0,self.m):
        #     #print(self.y[j,0])
        #     self.y_ob.append(self.y[j,0])
        #self.plotear()
    
    def change_W(self, error, x, y):
        nw = [0,0,0]
        for i in range(0,self.n):
            self.W[i] = self.W[i] + (self.theta * error * (y * (1 - y)) * x[i])
            
        

    def pw(self,x,w):
        pw_ = 0
        for i in range(0,self.n):
            pw_ += x[i] * w[i]

        #return sigmoid(pw_)
        return sigmoid(pw_)

    
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


    def barrido(self):
        y = -1
        x = -1
        while y <= 1:
            x=-1
            while x <= 1:
                
                pw_ = sigmoid(self.pw([1,x,y],self.W))
                #color = round(pw_, 2)

                if(pw_ >= 0.5): #Mayor a 0.5 - Clase 1
                    plt.plot(x,y, marker = "o", color = (0, pw_, 0)) # Verde
                else: # Clase 0                            R   G   B
                    plt.plot(x,y, marker = "o", color = (1-pw_, 0, 0)) # Rojo
                x+=0.1

            y+=0.1

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

def run():
    matriz = np.loadtxt('dataset_Perceptron.txt',delimiter = ',')
    em = 100
    theta = 0.1
    W = [random.random() for i in range(1,4)]
    err = 0.02

    #p = Perceptron_(W,matriz,theta,em)
    #p.iniciar()

    a = Adeline_(W,matriz,theta,em,err)
    a.iniciar()

if __name__ == '__main__':
    run()


