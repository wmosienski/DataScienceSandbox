import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('./heart.csv').dropna()

sns.displot(data['age'])
plt.show()

print("skew: " + str(data['age'].skew()))
print("kurt: " + str(data['age'].kurt()))

columns = data.columns.tolist()
columns.remove('output')

meta = pd.DataFrame(columns=columns)
meta.loc[len(meta.index)] = [abs(np.corrcoef(data[col], data['output'])[0,1]) for col in columns]

meta = meta.T.sort_values(0).T

meta.iloc[0].plot(kind='bar')
plt.show()

x = data.iloc[:, 0:-1]
y = data.iloc[:, -1:]

x_train, x_test, y_train, y_test = train_test_split(x, y)

for i in range(len(meta.columns)-2):
    j = i + 1
    my_x_train = x_train.drop(meta.columns[-j:].tolist(), axis=1)
    my_x_test = x_test.drop(meta.columns[-j:].tolist(), axis=1)

    print("columns: " + str(len(my_x_train.columns) - 1))

    model = RandomForestClassifier()

    model.fit(my_x_train, y_train.values.ravel())

    result = model.score(my_x_test, y_test)

    print("accuracy: " + str(result))
