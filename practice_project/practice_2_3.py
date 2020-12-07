#%%1
#모듈 임포트
import numpy as np
import matplotlib.pyplot as plt

#%%2
# 함수구현
#sigmoid 함수
def sigmoid(x):
    return 1/(1+np.exp(-x))

#%%3
#softmax
def softmax(x):
    e_x=np.exp(x)
    return e_x/np.sum(e_x)

#%%4
#네트워크 구조 정의
class ShallowNN:
    def __init__(self, num_input, num_hidden, num_output):
        self.W_h=np.zeros((num_hidden, num_input),dtype=np.float32)
        self.b_h=np.zeros((num_hidden,),dtype=np.float32)
        self.W_o=np.zeros((num_output, num_hidden),dtype=np.float32)
        self.b_h=np.zeros((num_output, ),dtype=np.float32)
    
    def __call__(self, x):
        h=sigmoid(np.matmul(self.W_h, x)+self.b_h)
        return softmax(np.matmul(self.W_o, h)+self.b_o)

#%%5
#데이터셋 가져오기, 정리하기
#Import and organize dataset
dataset=np.load('ch2_dataset.npz')
inputs=dataset['inputs']
labels=dataset['labels']

#%%6
#모델 만들기
model=ShallowNN(2, 128, 10)

#%%7
#사전에 학습된 파라미터 불러오기
weights=np.load('ch2_parameters.npz')
model.W_h=weights['W_h']
model.b_h=weights['b_h']
model.W_o  
#%%8
#모델 구동 및 결과 프린트


#%%9
#정답 클래스 스캐터 플랏


# %%
#출력 클래스 스캐터 플랏

