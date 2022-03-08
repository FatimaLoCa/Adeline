import random

import numpy as np

class Perceptron_:
    def __init__(self, W, matriz, theta, epochM):
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
        self.point_calculado=[]

    def iniciar(self):
        self.X = np.hstack((self.ones,self.X)) #agrega los -1
        self.n += 1
        done = False
        error=0
        em=0
        print("con -1",self.X)
        m=-(self.W[1]/self.W[2])
        b=(self.W[0]/self.W[2])
        calculados=[x2*m +b for x2 in self.X]
        self.point_calculado.append(calculados)
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
                    self.point_calculado.append(calculados)

            em += 1
            if em == self.epochM:
                return self.point_calculado,-1
                
        m=-(self.W[1]/self.W[2])
        b=(self.W[0]/self.W[2])
        calculados=[x2*m +b for x2 in self.X]
        return self.point_calculado
         
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


def run():
    pass