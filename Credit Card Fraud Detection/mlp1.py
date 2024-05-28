import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

# Read the dataset
dataset = pd.read_csv('cc2013.csv')
print(dataset['Class'].value_counts())

# no need id part
dataset = dataset.drop(labels=['Time'], axis=1)

y= dataset['Class']
x= dataset.drop(labels=['Class'], axis=1)

# Normalizing the data withour Class column
# norm='l2' means that the total sum of the squares of each element is 1
normalizer = Normalizer(norm='l2')
x1 = normalizer.fit_transform(x)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.20, random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.50, random_state=42)

# Creating the model
model = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=500, random_state=42)

# Creating the pipeline
pipeline = Pipeline([('model', model)])

# Cross Validation
scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring='accuracy')

# Mean Cross Validation Accuracy
mean_accuracy = scores.mean()

# Fitting the model
pipeline.fit(x_train, y_train)

# Predicting the Test set results
y_pred = pipeline.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Results Terminal Outputs
print("Model:", MLPClassifier())
print("Cross Validation Accuracy:", mean_accuracy)
print("Test Accuracy:", accuracy)
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

best_model = pipeline

# Confusion Matrix Visualization
LABELS = ['Normal', 'Fraud'] 
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix MLP1") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()