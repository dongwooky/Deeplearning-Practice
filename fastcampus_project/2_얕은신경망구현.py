import numpy as np
import matplotlib.pyplot as plt

# 함수구현
#sigmoid 함수

def sigmoid(x):
    return 1/(1+np.exp(-x))

#softmax

def softmax(x):
    e_x=exp(x)
    return e_x/np.sum(e_x)

#네트워크 구조 정의
class ShallowNN:
    def __init__(self,num_input,num_hidden,num_output):
        self.W_h=np.zeors((num_hidden,num_input),dtype=np.float32)
        self.b_h=np.zeros((num_hidden,),dtype=np.float32)
        self.W_o=np.zeors((num_output,num_hidden),dtype=np.float32)
        self.b_o=np.zeros((num_output,),dtype=np.float32)
    def __call__(self,x):
        h=sigmoid(np.matmul(self.W_h,x)+self.b_h)
        return softmax(np.matmul(self.W_o,h)+self.b_o)