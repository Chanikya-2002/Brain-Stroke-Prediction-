# -*- coding: utf-8 -*-
"""chatbot_work.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Zuur7GLjG3o0OjgotJ-mwi0HkgLVY6FJ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

df=pd.read_csv(r'/content/healthcare-dataset-stroke-data.csv')
df

df=df.drop('id',axis=1)

df.shape

df.info()

df.describe()

df.isnull().sum()

df["work_type"].unique()

df["gender"].unique()

df["age"].unique()

df["heart_disease"].unique()

df["ever_married"].unique()

df["Residence_type"].unique()

df["smoking_status"].unique()

df["stroke"].unique()

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['int', 'float'])

# Calculate correlation
correlation_matrix = numeric_columns.corr()

plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True)
plt.title('Heatmap of Correlations',fontsize=15)
plt.show()

df

df["age"].value_counts()

df.info()

df.shape

df.isnull().sum()

# Replace missing values in a specific column with the mean of that column
feature_column = 'bmi'
df[feature_column] = df[feature_column].fillna(df[feature_column].median())

df

import pandas as pd

# Assuming df is your DataFrame containing both numerical and categorical columns
numerical_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
text_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Create new DataFrames containing only numerical and text columns
df_numerical = df[numerical_columns]
df_text = df[text_columns]

# Display the numerical and text DataFrames
print("Numerical Columns:")
print(df_numerical.head())

print("\nText Columns:")
print(df_text.head())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import joblib
objList = ['gender','ever_married','work_type','Residence_type','smoking_status']
for feat in objList:
    df[feat] = le.fit_transform(df[feat].astype(str))
print (df.info())
joblib.dump(le, 'label_encoder.joblib')

y = df["stroke"]
sns.countplot(y)
target_temp = df.stroke.value_counts()

print(target_temp)

df["stroke"].value_counts()

print("Percentage of patience without stroke problems: "+str(round(target_temp[0]*100/299,2)))
print("Percentage of patience with stroke problem : "+str(round(target_temp[1]*100/299,2)))

df["gender"].unique()

countFemale = len(df[df.gender == 0])
countMale = len(df[df.gender == 1])
print("Percentage of Female Patients:{:.2f}%".format((countFemale)/(len(df.gender))*100))
print("Percentage of Male Patients:{:.2f}%".format((countMale)/(len(df.gender))*100))

pd.crosstab(df.age,df.stroke).plot(kind="bar",figsize=(20,6))
plt.title('Stroke Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('StrokeDiseaseAndAges.png')
plt.show()

pd.crosstab(df.gender,df.stroke).plot(kind="bar",figsize=(20,10),color=['blue','#AA1111','green','yellow','black' ])
plt.title('Stroke Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["No Stroke", "Stroke"])
plt.ylabel('Frequency')
plt.show()

df["Residence_type"].unique()

predictors = df.drop("stroke",axis=1)
target = df["stroke"]

target.value_counts()

from imblearn.over_sampling import SMOTE

print("Before OverSampling, counts of label '0': {}".format(sum(target == 0)))
print("Before OverSampling, counts of label '1': {}".format(sum(target == 1)))


# import SMOTE module from imblearn library
# pip install imblearn (if you don't have imblearn in your system)

sm = SMOTE(random_state = 65)
predictors_res, target_res = sm.fit_resample(predictors,target.ravel())

print('After OverSampling, the shape of train_X: {}'.format(predictors_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(target_res.shape))

print("After OverSampling, counts of label '0': {}".format(sum(target_res == 0)))
print("After OverSampling, counts of label '1': {}".format(sum(target_res == 1)))

X = predictors_res
y = target_res

# Split the data into training (80%), validation (10%), and testing (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(X_test)

# Select numerical columns
numerical_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# Select categorical columns
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Standardize numerical columns for training, validation, and testing sets
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

pip install tensorflow-addons

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Embedding, Input, concatenate, Reshape, BatchNormalization, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import MultiHeadAttention

# Assume predictors_res and target_res are your feature and target datasets

# Split the data into features (predictors_res) and target (target_res)


# Define the inputs
input_numerical = Input(shape=(X_train[numerical_columns].shape[1],))
input_categorical = Input(shape=(X_train[categorical_columns].shape[1],))

# Embedding layer for categorical features
embedding_layer = Embedding(input_dim=5, output_dim=5, input_length=4)(input_categorical)
flatten_embedding = Flatten()(embedding_layer)

# Convolutional and LSTM layers
x = Reshape((X_train[numerical_columns].shape[1], 1))(input_numerical)
x = Conv1D(filters=1024, kernel_size=9, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

# Additional Convolutional Layers
x = Conv1D(filters=512, kernel_size=8, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

x = Conv1D(filters=256, kernel_size=7, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

x = Conv1D(filters=128, kernel_size=6, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

x = Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

# Regular Convolutional Layer
x = Conv1D(filters=512, kernel_size=8, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

# LSTM Layers
x = LSTM(1024, return_sequences=True)(x)
x = Dropout(0.25)(x)
x = LSTM(512, return_sequences=True)(x)
x = Dropout(0.25)(x)

# Transformer-like layer for numerical data
att_output = MultiHeadAttention(num_heads=4, key_dim=2)(x, x)
transformer_output = Flatten()(att_output)

# Concatenate the output of embedding and transformer layers
concatenated_layer = concatenate([flatten_embedding, transformer_output])

# Dense layers for classification
dense_layer1 = Dense(512, activation='relu')(concatenated_layer)
dense_layer1 = Dropout(0.5)(dense_layer1)
dense_layer2 = Dense(256, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.5)(dense_layer2)
output_layer = Dense(1, activation='sigmoid')(dense_layer2)

# Define the model
model = Model(inputs=[input_numerical, input_categorical], outputs=output_layer)

# Compile the model with Adamax optimizer
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    [X_train[numerical_columns], X_train[categorical_columns]],
    y_train,
    epochs=100,
    batch_size=256,
    validation_data=([X_val[numerical_columns], X_val[categorical_columns]], y_val)
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate([X_test[numerical_columns], X_test[categorical_columns]], y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy}')

test_loss

# Accept user inputs for the features
age = float(input("Enter age: "))
gender = int(input("Enter gender (0 for female, 1 for male): "))
hypertension = int(input("Enter hypertension status (0 for No, 1 for Yes): "))
heart_disease = int(input("Enter heart disease status (0 for No, 1 for Yes): "))
ever_married = int(input("Enter marital status (0 for No, 1 for Yes): "))
work_type = int(input("Enter work type (0 for Private, 1 for Self-employed, 2 for Govt_job, 3 for Children, 4 for Never_worked): "))
residence_type = int(input("Enter residence type (0 for Urban, 1 for Rural): "))
avg_glucose_level = float(input("Enter average glucose level: "))
bmi = float(input("Enter BMI: "))
smoking_status = int(input("Enter smoking status (0 for Unknown, 1 for formerly smoked, 2 for never smoked, 3 for smokes): "))

# Preprocess the user inputs
X_user_numerical = scaler.transform([[age, hypertension, heart_disease, avg_glucose_level, bmi]])
X_user_categorical = np.array([[gender, ever_married, work_type, residence_type, smoking_status]])

# Make prediction
prediction = model.predict([X_user_numerical, X_user_categorical])

# Convert prediction to human-readable format
if prediction[0] >= 0.5:
    result = "likely to have a stroke"
else:
    result = "unlikely to have a stroke"

print("Based on the provided information, the person is", result)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have trained the model and obtained predictions on test data
y_pred = model.predict([X_test[numerical_columns], X_test[categorical_columns]])
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (Test Data)')
plt.show()

# Calculate other evaluation metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
from sklearn.metrics import classification_report


# Generate classification report
report = classification_report(y_test, y_pred_binary)

print("Classification Report:")
print(report)

# Assuming you have trained the model and obtained predictions on test data
loss, accuracy = model.evaluate([X_test[numerical_columns], X_test[categorical_columns]], y_test)

print(f'Loss on Test Data: {loss:.4f}')

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Embedding, Input, concatenate, Reshape, BatchNormalization, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import MultiHeadAttention
import numpy as np

# Define the inputs
input_numerical = Input(shape=(X_train_numerical.shape[1],))
input_categorical = Input(shape=(X_train_categorical.shape[1],))

# Embedding layer for categorical features
embedding_layer = Embedding(input_dim=5, output_dim=5, input_length=4)(input_categorical)
flatten_embedding = Flatten()(embedding_layer)

# Convolutional and LSTM layers
x = Reshape((X_train_numerical.shape[1], 1))(input_numerical)
x = Conv1D(filters=1024, kernel_size=9, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

# Additional Convolutional Layers
x = Conv1D(filters=512, kernel_size=8, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

x = Conv1D(filters=512, kernel_size=8, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

x = Conv1D(filters=256, kernel_size=7, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

x = Conv1D(filters=128, kernel_size=6, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

x = Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

# Regular Convolutional Layer
x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)

# LSTM Layers
x = LSTM(1024, return_sequences=True)(x)
x = Dropout(0.25)(x)
x = LSTM(512, return_sequences=True)(x)
x = Dropout(0.25)(x)

# Transformer-like layer for numerical data
att_output = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
transformer_output = Flatten()(att_output)

# Concatenate the output of embedding and transformer layers
concatenated_layer = concatenate([flatten_embedding, transformer_output])

# Dense layers for classification
dense_layer1 = Dense(512, activation='relu')(concatenated_layer)
dense_layer1 = Dropout(0.5)(dense_layer1)
dense_layer2 = Dense(256, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.5)(dense_layer2)
output_layer = Dense(1, activation='sigmoid')(dense_layer2)

# Define the model
model = Model(inputs=[input_numerical, input_categorical], outputs=output_layer)

# Compile the model with Nadam optimizer
optimizer =Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit([X_train_numerical, X_train_categorical], y_train, epochs=150, batch_size=256, validation_split=0.2)

# Evaluate the model
accuracy = model.evaluate([X_test_numerical, X_test_categorical], y_test)[1]
print("Test Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have your test data ready: X_test_numerical, X_test_categorical, and y_test

# Predict on test data
y_pred = model.predict([X_test_numerical, X_test_categorical])
y_pred_classes = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={'fontsize': 15}, linewidths=0.5, linecolor='black')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(df_categorical):
    encoder = LabelEncoder()
    df_encoded = df_categorical.copy()  # Create a copy of the DataFrame

    # Encode each categorical column in the DataFrame
    for column in df_categorical.columns:
        df_encoded[column] = encoder.fit_transform(df_categorical[column])

    return df_encoded

import matplotlib.pyplot as plt

def plot_accuracy(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

# Call the function with the history object
plot_accuracy(history)

pr=model.predict([X_test[numerical_columns], X_test[categorical_columns]])

import pickle
pickle.dump(pr,open("model.pkl","wb"))
