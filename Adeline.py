import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from Perceptron import Perceptron_
#import rhinoscriptsytnax as rs



def sigmoid(z):
    return 1 / (1 + math.exp(-z))


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
        
        
        self.y = np.zeros((self.m,1)) # Vector para y obtenidas, calculadas con pw

        self.ones= np.ones((self.m,1)) * -1 #Vector de -1
        self.y_ob = []

    def iniciar(self):
        self.X = np.hstack((self.ones,self.X)) #agrega los -1
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
                pw_ = self.pw(self.X[i],self.W) #Dentro de pw se hace la sigmoide 

                #have = self.fx(pw_) # f(y) = f(pw)}
                have = pw_
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
            nw[i] = self.W[i] + (self.theta * error * (sigmoid(y) * (1 - sigmoid(y))) * x[i])
        self.W = nw
        

    def pw(self,x,w):
        pw_ = 0
        for i in range(0,self.n):
            pw_ += w[i] * x[i]

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
        #print(have)
        #print(have2)
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

        plt.plot(v_clase0[:,1],v_clase0[:,2] , 'ro') # Clase 0
        plt.plot(v_clase1[:,1],v_clase1[:,2] , 'bo') # Clase 1


    def degradado(self):
        pass
            
    
def run():
    
    matriz = np.loadtxt('dataset_Perceptron.txt',delimiter = ',')
    em = 100
    theta = 0.1
    W = [random.random() for i in range(1,4)]
    print(W)
    err = 0.1

    #p = Perceptron_(W,matriz,theta,em)
    #p.iniciar()

    a = Adeline_(W,matriz,theta,em,err)
    a.iniciar()

if __name__ == '__main__':
    run()


