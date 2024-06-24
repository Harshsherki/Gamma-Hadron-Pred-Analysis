# MAGIC Gamma Telescope Data Classification

This project uses machine learning techniques to classify data from the MAGIC Gamma Telescope. The dataset consists of 11 features and a binary class label indicating whether the event is a gamma signal (1) or a hadron signal (0). Various classification models are applied, including K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, and Support Vector Machines (SVM).

## Dataset

The dataset used in this project contains the following columns:

- `fLength`: Major axis length
- `fWidth`: Minor axis length
- `fSize`: Size (area)
- `fConc`: Conc. of lights
- `fConc1`: Conc. of lights in 1st region
- `fAsym`: Asymmetry
- `fM3Long`: 3rd Root of Long. Distance
- `fM3Trans`: 3rd Root of Transverse Distance
- `fAlpha`: Angle
- `fDist`: Distance from the camera center
- `class`: Class label (1 for gamma, 0 for hadron)

## Requirements

- numpy
- pandas
- matplotlib
- scikit-learn
- imbalanced-learn


## Data Preprocessing

The data is first read into a pandas DataFrame, and the class labels are converted to binary values (1 for gamma, 0 for hadron).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

columns = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=columns)
df["class"] = (df["class"] == 'g').astype(int)
```

## Exploratory Data Analysis

Histograms are plotted to visualize the distribution of features for each class.

```python
for col in columns[:-1]:
    plt.hist(df[df["class"] == 1][col], label="gamma", density=True, alpha=0.9, color="blue")
    plt.hist(df[df["class"] == 0][col], label="hadron", density=True, alpha=0.5, color="yellow")
    plt.title(col)
    plt.ylabel("Probability")
    plt.xlabel(col)
    plt.legend()
    plt.show()
```

## Data Preparation

The data is split into training, validation, and test sets. The features are standardized, and oversampling is applied to the training set to handle class imbalance.

```python
def scale_dataset(dataframe, oversampler=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversampler:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))
    return data, x, y

train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

train, x_train, y_train = scale_dataset(train, oversampler=True)
valid, x_valid, y_valid = scale_dataset(valid, oversampler=False)
test, x_test, y_test = scale_dataset(test, oversampler=False)
```

## Model Training and Evaluation

### K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
y_pred = knn_model.predict(x_train)
print(classification_report(y_pred, y_train))
```

### Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
y_pred = nb_model.predict(x_train)
print(classification_report(y_pred, y_train))
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_train)
print(classification_report(y_pred, y_train))
```

### Support Vector Machines (SVM)

```python
from sklearn.svm import SVC
from sklearn.metrics import classification_report

svm_model = SVC()
svm_model.fit(x_train, y_train)
y_pred = svm_model.predict(x_train)
print(classification_report(y_pred, y_train))
```

## Results

- **KNN Model**:
  - Precision: 0.89 (class 0), 0.90 (class 1)
  - Recall: 0.90 (class 0), 0.89 (class 1)
  - F1-score: 0.89 (class 0), 0.90 (class 1)
  - Accuracy: 0.90

- **Naive Bayes Model**:
  - Precision: 0.41 (class 0), 0.90 (class 1)
  - Recall: 0.80 (class 0), 0.60 (class 1)
  - F1-score: 0.54 (class 0), 0.72 (class 1)
  - Accuracy: 0.65

- **Logistic Regression Model**:
  - Precision: 0.72 (class 0) , 0.83 (class 1)
  - Recall: 0.81 (class 0) , 0.75 (class 1)
  - F1-score: 0.76 (class 0) , 0.79 (class 1)
  - Accuracy: 0.78

- **SVM Model**:
  - Precision: 0.81 (class 0) , 0.90 (class 1)
  - Recall: 0.89 (class 0) , 0.83 (class 1)
  - F1-score: 0.85 (class 0) , 0.86 (class 1)
  - Accuracy: 0.86

## Conclusion

The SVM model performs the best with an accuracy of 86%, followed by KNN, Logistic Regression, and Naive Bayes. Future improvements could include trying more complex models and feature engineering to improve the prediction accuracy.

