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


#%%5
#데이터셋 가져오기, 정리하기
#Import and organize dataset


#%%6
#모델 만들기


#%%7
#사전에 학습된 파라미터 불러오기


#%%8
#모델 구동 및 결과 프린트


#%%9
#정답 클래스 스캐터 플랏


# %%
#출력 클래스 스캐터 플랏

