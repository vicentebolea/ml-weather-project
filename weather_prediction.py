import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing as pre, svm
from sklearn.cross_validation import train_test_split
df = pd.read_csv("data_Set_pro/ulsan_final.csv")
X=np.array(df.drop(['TEMPERATURE','dateTime','TIME'], 1))
y= df['TEMPERATURE']

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)
scaler = pre.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("training model.....")
SVR_model = svm.SVR(kernel='rbf',C=100,gamma=.1).fit(X_train_scaled,y_train)
print("prediction of model...")
predict_y_array = SVR_model.predict(X_test_scaled)
score=SVR_model.score(X_test_scaled,y_test)
print(score)

print(predict_y_array[0:10], np.array(y_test[0:10]))
print(len(y_test))
print(len(predict_y_array))
plt.plot( np.array(y_test[0:100]), color='g')
plt.plot( predict_y_array[0:100],color='r')
plt.xlabel('Datetime')
plt.ylabel('Temprature')
plt.title('Temprature forecasting')
plt.show()


