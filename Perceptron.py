
import numpy as np
import matplotlib.pyplot as plt

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
        self.ones= np.ones((self.m,1)) * -1
        self.y_ob = []
        #print(self.X[:,0])
        print(self.m)
        print(self.n)
        #self.plotear()

    def iniciar(self):
        self.X = np.hstack((self.ones,self.X)) #agrega los -1
        self.n += 1
        print(self.X)
        done = False
        error=0
        em=0
        #print(self.epochM)
        while done == False and em < self.epochM: #mientras no se ha terminado y no se cumplan las epocas
            done = True
            #print(em)
            for i in range(0,self.m): 
                have = self.pw(self.X[i],self.W)
                error = self.Y[i] - have
                #print('i= '+str(i)+'deseo: '+str(self.Y[i])+' obtuve: '+ str(have))
                if error != 0:
                    #print('entré')
                    done = False
                    self.change_W(error, self.X[i])
            
            em += 1

        self.mostrar()
        self.calcular_Y_ob()
        print(self.y_ob)
        print(self.Y)

    def calcular_Y_ob(self):
        for i in range(0,self.m):
            have = self.pw(self.X[i],self.W)
            self.y_ob.append(have)

    def change_W(self, error, x):
        nw = [0,0,0]
        for i in range(0,self.n):
            nw[i] = self.W[i] + (self.theta * error * x[i])
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

    def plotear(self):
        #plt.plot(self.X[:,0],self.X[:,1] , 'ro')
        clase0 = []
        clase1 = []
        for i in range(0,self.m):
            if self.Y[i] == 0:
                clase0.append(self.X[i])
            else:
                clase1.append(self.X[i])
        v_clase0 = np.array(clase0)
        v_clase1 = np.array(clase1)
        plt.plot(v_clase0[:,0],v_clase0[:,1] , 'ro')
        plt.plot(v_clase1[:,0],v_clase1[:,1] , 'bo')
        plt.axis([-1, 1, -1, 1])
        plt.show()
