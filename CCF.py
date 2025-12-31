import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

try:
    credit_card_data = pd.read_csv('creditcard.csv')
    print(" Data Loaded Successfully")
except FileNotFoundError:
    print('creditcard.csv file not found')

print('\n Data summary')
print(credit_card_data.info())

if credit_card_data.isnull().sum().sum() > 0 :
    print('Missing values is found , filling it with ffill')
    credit_card_data.fillna(method ='ffill', inplace = True)

plt.figure(figsize = (10,6))
sns.countplot( x = 'Class', data = credit_card_data, palette = 'Set2')
plt.title('Fraud vs Non Fraud detection')
plt.xlabel('Transaction class ( 0 = Non fraud) (1 = fraud)')
plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
plt.show()

plt.figure(figsize = (10,6))
corr = credit_card_data.corr()

sorted_corr = corr['Class'].abs().sort_values(ascending = False)

sns.heatmap(corr[sorted_corr.index][:20], cmap = 'coolwarm', annot = False, linewidth = 0.5)
plt.title('Top correlation haetmap')
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data = pd.read_csv('creditcard.csv')

credit_card_data.head()

credit_card_data.info()

credit_card_data.isnull().sum()

credit_card_data['Class'].value_counts()

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

legit.Amount.describe()

fraud.Amount.describe()

credit_card_data.groupby('Class').mean()

legit_sample = legit.sample(n = 492)

new_dataset = pd.concat([legit_sample, fraud], axis = 0)
new_dataset.head()

new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

X = new_dataset.drop(columns = 'Class', axis = 1)
y = new_dataset['Class']

print(X)

print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)

X_train_prediction= model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, y_train)

print("Accuracy on Training Data :", train_data_accuracy)

X_test_prediction= model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print("Accuracy on Testing Data :", test_data_accuracy)

import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('creditcard.csv')
df.describe()
df.info()

from sklearn.model_selection import train_test_split
X = df.drop('Class', axis = 1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)
predictions

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

from sklearn import tree

plt.figure(figsize = (20,25))

tree.plot_tree( dtree,
                feature_names = X.columns,
                class_names = ['Class-1','Class-0'],
                rounded = True,
                filled = True,
                proportion = True)

plt.show()

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

credit_card_data = pd.read_csv('creditcard.csv')

credit_card_data.describe()
credit_card_data.isnull().sum()
credit_card_data.info()

credit_card_data.hist(figsize = (20,20))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(credit_card_data.drop(['Class'], axis = 1)))
y = credit_card_data.Class

x.head()

y.head()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(x_train, y_train)
pred = knn.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize = (10,8))
plt.plot( range(1,40),
          error_rate,
          color ='blue',
          linestyle = 'dashed',
          marker = 'o',
          markerfacecolor = 'red',
          markersize = 10)
plt.title('Error rate vs k value')
plt.show()


# Model comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)

credit_card_data = pd.read_csv('creditcard.csv')
print(' Data loaded successfully')

print(credit_card_data.shape)
credit_card_data['Class'].value_counts()

plt.figure(figsize = (15,10))
sns.heatmap(credit_card_data.corr(), cmap = 'coolwarm', annot = False)

plt.title("Correlation heatmap")
plt.show()

x = credit_card_data.drop('Class', axis = 1)
y = credit_card_data['Class']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42, stratify = y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model1 = LogisticRegression()
model2 = KNeighborsClassifier()
model3 = DecisionTreeClassifier()

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)

y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)

acc1 = accuracy_score(y_test, y_pred1)
prec1 = precision_score(y_test, y_pred1)
rec1 = recall_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1)

acc2 = accuracy_score(y_test, y_pred2)
prec2 = precision_score(y_test, y_pred2)
rec2 = recall_score(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2)

acc3 = accuracy_score(y_test, y_pred3)
prec3 = precision_score(y_test, y_pred3)
rec3 = recall_score(y_test, y_pred3)
f1_3 = f1_score(y_test, y_pred3)

results = pd.DataFrame({'Model' : ['Logistic regression','KNN','Decision tree'],
                        'Accuracy' : [acc1,acc2,acc3],
                        'Precision' : [prec1,prec2,prec3],
                        'Recall': [rec1,rec2,rec3],
                        'F1  score' : [f1_1,f1_2,f1_3]})

print('Model comparison')

print(results.sort_values(by = 'F1  score', ascending = False))


# Model deployment

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv('creditcard.csv')

x = data.drop('Class', axis = 1)
y = data['Class']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 2, stratify = y)

model_pipeline = Pipeline([('scaler', StandardScaler()), ('log_reg', LogisticRegression(solver = 'lbfgs'))])

model_pipeline.fit(x_train, y_train)

joblib.dump(model_pipeline , 'fraud_detection_model.pkl')

print('Model saved as a fraud_detection_model.pkl')

loaded_model = joblib.load('fraud_detection_model.pkl')

sample = x_test.iloc[0:1]

prediction = loaded_model.predict(sample)

print('Prediction ( 0 : Not fraud, 1 : Fraud) :', prediction[0])

accuracy = loaded_model.score(x_test, y_test)

print(f'Model accuracy on test data: {accuracy : .2f}')

import joblib
import numpy as np
import ipywidgets as widgets
from IPython.display import display

from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

joblib.dump(scaler, 'scaler.pkl')

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)

joblib.dump(model,'fraud_detection_model.pkl')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

joblib.dump(scaler, 'scaler.pkl')


model = LogisticRegression()
model.fit(x_train, y_train)

joblib.dump(model,'fraud_detection_model.pkl')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('creditcard.csv')

selected_features = ['Amount','V14','V10','V12']
x = df[selected_features]
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(scaler, 'scaler.pkl')

print('Model and scaler saved for 4 features')

import joblib
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output

model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

output = widgets.Output()

def predict_new_transactions(amount,v14,v10,v12):
    new_data = np.array([[amount,v14,v10,v12]])
    new_data_scaled = scalar.transform(new_data)
    prediction = model.predict(new_data_scaled)
    with output:
        clear_output()
        print('Prediction ( 0 : Not fraud, 1 : Fraud) :', prediction[0])

amount_input = widgets.FloatText(description = 'Amount')
v14_input = widgets.FloatText(description = 'V14')
v10_input = widgets.FloatText(description = 'V10')
v12_input = widgets.FloatText(description = 'V12')

predict_button = widgets.Button(description = 'Predict')

def on_click(b):
    predict_new_transaction( amount_input.value, v14_input.value, v10_input.value, v12_input.value)

predict_button.on_click(on_click)

display(amount_input, v14_input, v10_input, v12_input, predict_button, output)
