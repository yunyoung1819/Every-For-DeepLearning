# 딥러닝을 구동하는데 필요한 케라스 함수를 불러온다
from keras.models import Sequential
from keras.layers import Dense

# 필요한 라이브러리
import numpy
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 준비된 수술 환자 데이터
Data_set = numpy.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장
X = Data_set[:,0:17]
Y = Data_set[:,17]

# 딥러닝 모델 설정
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, epochs=30, batch_size=10)

# 결과 출력
print("\n Accuracy:: %.4f" % (model.evaluate(X,Y)[1]))




