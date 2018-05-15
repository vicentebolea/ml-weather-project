import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.externals import joblib
from sklearn import preprocessing as pre, svm
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as p

# Needed for plotting using Linux
p.switch_backend('TKAgg')  

# Remove annoying warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

df = pd.read_csv("input/ulsan.csv")
x = np.array(df.drop(['TEMPERATURE', 'dateTime', 'TIME'], 1))
y = df['TEMPERATURE']

# split into a training and testing set
x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.25)

scaler = pre.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


print("Loading model.....")
# Do not recompute if possible
try:
    SVR_model = joblib.load('model.pkl')

except:
    print("Failed, training model.....")
    SVR_model = svm.SVR(kernel='rbf',C=100,gamma=.1).fit(x_train_scaled, y_train)
    joblib.dump(SVR_model, 'model.pkl')

print("prediction of model...")

predict_y_array = SVR_model.predict(x_test_scaled)
score = SVR_model.score(x_test_scaled,y_test)

print(score)
print(predict_y_array[0:10], np.array(y_test[0:10]))

print("Plotting the model")

plt.plot(np.array(y_test[0:100]), color='g')
plt.plot(predict_y_array[0:100], color='r')

plt.xlabel('Datetime')
plt.ylabel('Temprature')
plt.title('Temprature forecasting')
plt.show()


