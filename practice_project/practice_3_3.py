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
        super(MyModel, self).__init__(self)

        self.d1=tf.layers.Dense(128, input_dim=2, activation='sigmoid')
        self.d2=tf.layers.Dense(10, activation='softmax')

    def __call__(self, inputs):
        x=self.d1(inputs)
        return self.d2(x)

#%%4
#학습 루프 정의
def training(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions=model(inputs)
        loss=loss_object(predictions, labels)

    gradients=tape.Gradient_Object(loss, model.trainable_variables)
    optimizer(gradients, model.trainable_variables)

    train_loss(loss)
    train_accuracy(labels, predictions)

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
