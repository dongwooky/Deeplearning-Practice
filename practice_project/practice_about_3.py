#%%1
#모듈 임포트
import tensorflow as tf
import numpy as np
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
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_metric):
    with tf.GradientTape() as tape:
        predictions=model(inputs)
        loss=loss_object(labels, predictions)
    gradients=tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_metric(labels, predictions)

#%%5
#데이터셋 생성, 전처리
np.random.seed(0)

pts=list()
labels=list()

center_pts=np.random.uniform(-8.0,8.0,(10,2))
for label, center_pt in enumerate(center_pts):
    for _ in range(100):
        pts.append(center_pt+np.random.randn(*center_pt.shape))
        labels.append(label)

pts=np.stack(pts, axis=0).astype(np.float32)
labels=np.stack(labels,axis=0)
        
train_ds=tf.data.Dataset.from_tensor_slices((pts,labels)).shuffle(1000).batch(32)
#%%6
#모델 생성
model=MyModel()
#%%7
#손실 함수 및 최적화 알고리즘 설정
#CrossEntropy, Adam Optimizer
loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
optimizer=tf.keras.optimizers.Adam()
#%%8
#평가 지표 설정
#Accuracy

#%%
#학습 루프

#%%
#데이터셋 및 학습 파라미터 저장
