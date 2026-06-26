import numpy as np

class kalmanFilter:
    def __init__(self,x,y): # en başta hızlar vx ve vy 0 sayılıyor 
        self.state = np.array([x,y,0,0], dtype=float) # state = [x,y,vx,vy]
        self.F = np.array([[1,0,1,0],
                           [0,1,0,1],
                           [0,0,1,0],
                           [0,0,0,1]], dtype=float) # state transition matrix
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=float) # measurement matrix
        
        self.P = np.eye(4)*1000
        self.R = np.eye(2)*10
        self.Q = np.eye(4)*0.1

    def predict(self):
        self.state =self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q  # F.T kısmı transpose kısmı .T transpose etmek
                                                    # yani matrisin sütunu satır satırı sütun oluyor
                                                    # P_yeni = F × P × F.T + Q  KALMAN MATEMATİĞİ
        return self.state[:2] # sadece x , y döndürür     
    
    def update(self,x,y):
        z = np.array([x,y] , dtype=float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ (z - self.H @ self.state)  # innovation denir bu kısma
        
        self.P = (np.eye(4) - K @ self.H) @ self.P                     
        
        return self.state[:2]  

