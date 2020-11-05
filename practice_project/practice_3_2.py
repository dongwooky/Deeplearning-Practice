#%%1
#모듈 임포트
import numpy as np
import tensorflow as tf

#%%2
#하이퍼 파라미터 설정
EPOCHS=1000

#%%3
#네트워크 구조 정의
#얇은 신경망
#입력 계층 : 2, 은닉 계층 : 128(Sigmoid activation), 출력 계층 : 10(Softmax activation)
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1=tf.keras.layers.Dense(128, input_dim=2, activation='sigmoid')
        self.dense2=tf.keras.layers.Dense(10, activation='softmax')

    def __call__(self, inputs, training=None, mask=None):
        x=self.dense1(inputs)
        return self.dense2(x)

#%%4
#학습 루프 정의
@tf.function

#%%5
#데이터셋 생성, 전처리

#%%6
#모델 생성

#%%7
#손실 함수 및 최적화 알고리즘 설정
#CrossEntropy, Adam Optimizer

#%%8
#평가 지표 설정
#Accuracy

#%%
#학습 루프

#%%
#데이터셋 및 학습 파라미터 저장
