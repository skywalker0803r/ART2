import numpy as np
import warnings
import math
import matplotlib.pyplot as plt

class ART2:
    def __init__(self,rho=0.997):#user only change rho
        self.a = 10
        self.b = 10
        self.c = 0.3
        self.d = 0.75
        self.theta = 0.1
        self.rho=rho
        self.e = 1E-8
    
    def Schema_ART2_F1(self,I, F2_feedback, a, b, theta, e, m):
        u = np.zeros((1,m))
        p = np.zeros((1,m))
        temp = np.ones((1,m))
        while np.sum(abs(temp-u)) >= 0.001:
            temp = u[:]
            w = I + a * u
            p = u + F2_feedback
            x = w / (e + np.linalg.norm(w, 2))
            q = p / (e + np.linalg.norm(p, 2))
            x[x<theta] = 0 # 反映 f(x(k))
            q[q<theta] = 0 # 反映 f(q(k))
            v = x + b * q
            u = v /(e + np.linalg.norm(v, 2))
        return u,p
    
    def ART_Process(self,I, NumNeurons, a, b, c, d, WF1_F2, WF2_F1, rho, theta, e):
        y = np.zeros(NumNeurons)
        Counter = 1
        F2_feedback = 0
        m = len(I)
        while Counter > 0:
            u, p = self.Schema_ART2_F1(I, F2_feedback, a, b, theta, e, m)
            y[abs(y+1)>e] = (np.dot(WF1_F2[list(abs(y + 1) > e), :], np.transpose(p))).flatten()
            maxV = np.max(y)
            Js = np.where(y == maxV)
            J = Js[0][0]
            F2_feedback = d * np.transpose(WF2_F1[:, J])
            p = u + F2_feedback
            r = (u + c * p) / (e + np.linalg.norm(u, 2) + c * np.linalg.norm(p, 2))
            R = np.linalg.norm(r, 2)
            if R < rho - e:
                y[J]= -1
                Counter = 1
            elif Counter == 1:
                Counter += 1
            else:
                WF2_F1[:, J] = np.transpose((u / (1 - d))[0])
                WF1_F2[J, :] = (u / (1 - d))[0];
                Counter = 0
        return u,J
    
    def fit_transform(self,X):
        '''
        INPUT:X(m,n)dim-DataFrame
        output:each sample's label
        '''
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        theta = self.theta
        rho = self.rho
        e = self.e
        X = np.array(X)
        m, n = X.shape
        output = m #sample_num
        #根據特徵數量計算兩個weight(initital)
        WF1_F2 = np.dot(np.ones((output,n)),0.5/((1-d)*math.sqrt(n)))
        WF2_F1 = np.zeros((n, output))
        #sample len zero list 
        w = np.zeros(m)
        #for each sample tag label
        for j in range(m):
            Input = X[j,:]#one sample
            #J is the one sample's label
            u,J = self.ART_Process(Input, output,a,b,c,d,WF1_F2,WF2_F1, rho,theta,e)
            #根據u&d計算兩個weight(update)
            WF2_F1[:,J] = np.transpose((u/(1-d))[0])
            WF1_F2[J,:] = (u/(1-d))[0]
            #for each sample tag label
            w[j] = J
        #SAVE Weight
        self.WF2_F1 = WF2_F1
        self.WF1_F2 = WF1_F2
        self.NumNeurons = m
        return w

    def fit(self,X):
        '''
        input:X(m,n)dim-DataFrame
        return:2 weight WF2_F1 & WF1_F2
        '''
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        theta = self.theta
        rho = self.rho
        e = self.e
        X = np.array(X)
        m, n = X.shape
        output = m #sample_num
        #根據特徵數量計算兩個weight(initital)
        WF1_F2 = np.dot(np.ones((output,n)),0.5/((1-d)*math.sqrt(n)))
        WF2_F1 = np.zeros((n, output))
        #sample len zero list 
        w = np.zeros(m)
        #for each sample tag label
        for j in range(m):
            Input = X[j,:]#one sample
            #J is the one sample's label
            u,J = self.ART_Process(Input, output,a,b,c,d,WF1_F2,WF2_F1, rho,theta,e)
            #根據u&d計算兩個weight(update)
            WF2_F1[:,J] = np.transpose((u/(1-d))[0])
            WF1_F2[J,:] = (u/(1-d))[0]
            #for each sample tag label
            w[j] = J
        #SAVE Weight
        self.WF2_F1 = WF2_F1
        self.WF1_F2 = WF1_F2
        self.NumNeurons = m
        return WF2_F1,WF1_F2

    def transform(self,X):
        '''
        INPUT:X(m,n)-dim-DataFrame
        return: each sample's label
        '''
        X= np.array(X)
        m,n = X.shape
        #output = m
        output = self.NumNeurons 
        #load weight
        WF1_F2 = self.WF1_F2
        WF2_F1 = self.WF2_F1
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        rho = self.rho
        theta = self.theta
        e = self.e
        #sample len zero list
        w = np.zeros(m)
        #for each sample tag label
        for j in range(m):
            #the one sample
            Input = X[j,:]
            #J is the one sample's label
            u,J = self.ART_Process(Input,output,a,b,c,d,WF1_F2,WF2_F1,rho,theta,e)
            #for each sample tag label
            w[j] = J
        return w
