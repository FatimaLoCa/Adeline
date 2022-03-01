import os
import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(z):
    return 1 / (1+math.exp(-z))


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
        
        #print(self.m)
        #print(self.n)
        #self.plotear()

    def iniciar(self):
        self.X = np.hstack((self.ones,self.X)) #agrega los -1
        self.n += 1
        error=0
        errorMed = 0
        em = 0 #Contador de epocas
        
        while errorMed == self.errorMin and em < self.epochM: #mientras no se ha terminado y no se cumplan las epocas
            errorMed = 0
            for i in range(0,self.m): 
                have = self.pw(self.X[i],self.W)
                self.y[i] = have
                error = self.Y[i] - have
                errorMed += error
                #print('i= '+str(i)+'deseo: '+str(self.Y[i])+' obtuve: '+ str(have))
                
                self.change_W(error, self.X[i], self.y[i])
            
            errorMed = error/self.m
            em += 1

        #self.mostrar()
        #self.calcular_Y_ob()
        print(self.y)
        #print(self.Y)
        self.plotear()

    
    def calcular_Y_ob(self):
        for i in range(0,self.m):
            have = self.y[i]
            if have >= 0 :
                self.y_ob.append(1)
            else:
                self.y_ob.append(0)
    
    
    def change_W(self, error, x, y):
        nw = [0,0,0]
        for i in range(0,self.n):
            nw[i] = self.W[i] + (self.theta * error * (sigmoid(y) * (1 - sigmoid(y))) * x[i])
        self.W = nw
        

    def pw(self,x,w):
        pw_ = 0
        for i in range(0,self.n):
            pw_ += w[i] * x[i]

        return sigmoid(pw_)
			
    def mostrar(self):
        print('W0 = %4f'% self.W[0]) 
        print('w1 = %4f'% self.W[1])
        print('W2 = %4f'% self.W[2])

    def plotear(self):
        for i in range(0,self.m):
            plt.plot(self.X[i][1],self.X[i][2], marker = "*", color = (0.8, 0.2, 0.5))

        plt.axis([-1, 1, -1, 1])
        plt.show()
    
def run():
    
    matriz = np.loadtxt('dataset_Perceptron.txt',delimiter = ',')
    em = 400
    theta = 0.1
    W = [5,4,3]
    err = 0

    #p = Perceptron_(W,matriz,theta,em)
    #p.iniciar()

    a = Adeline_(W,matriz,theta,em,err)
    a.iniciar()

if __name__ == '__main__':
    run()