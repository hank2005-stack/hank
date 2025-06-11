# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:13:53 2025

@author: GIGABYTE
"""

import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
import random

np.random.seed(10)

IMG_SIZE = 150
X = [] 
Z = [ ]  

FLOWER_DIRS = {
    "daisy": "C:/Users/GIGABYTE/Desktop/flower_photos/daisy",
    "sunflower": "C:/Users/GIGABYTE/Desktop/flower_photos/sunflowers",
    "tulip": "C:/Users/GIGABYTE/Desktop/flower_photos/tulips",
    "dandelion": "C:/Users/GIGABYTE/Desktop/flower_photos/dandelion",
    "rose": "C:/Users/GIGABYTE/Desktop/flower_photos/roses"
}

def make_train_data(flower_type, DIR):
    global X, Z
    for filename in os.listdir(DIR):
        path = os.path.join(DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        Z.append(flower_type)  
    
for flower, path in FLOWER_DIRS.items():
    make_train_data(flower, path)

flower_counts = Counter(Z)
print("各類花卉圖片數量:", flower_counts)
print("總圖片數量:", len(X))

X_array = np.array(X, dtype="float32")  
X = X_array / 255.0 


encoder = LabelEncoder()
Z = encoder.fit_transform(Z)  
Z = to_categorical(Z, num_classes=len(FLOWER_DIRS))  


X_train, X_temp, Z_train, Z_temp = train_test_split(X, Z, test_size=0.3, random_state=42)
X_val, X_test, Z_val, Z_test = train_test_split(X_temp, Z_temp, test_size=1/3, random_state=42)

model = Sequential()
model.add(Input(shape=(IMG_SIZE,IMG_SIZE, 3)))
model.add(Conv2D(32, kernel_size=(5, 5), padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(5, activation="softmax"))

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, Z_train, epochs=13, batch_size=64, verbose=1, validation_data=(X_val, Z_val))

loss, accuracy = model.evaluate(X_train, Z_train, verbose=0)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Z_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))

loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(Z_test, axis=1)

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)


plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(FLOWER_DIRS.keys()), yticklabels=list(FLOWER_DIRS.keys()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

num_samples = 5 
random_indices = random.sample(range(len(X_test)), num_samples)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(random_indices):
    img = X_test[idx]
    true_label = list(FLOWER_DIRS.keys())[np.argmax(Z_test[idx])]
    pred_label = list(FLOWER_DIRS.keys())[np.argmax(model.predict(np.expand_dims(img, axis=0)))]

    plt.subplot(1, num_samples, i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True: {true_label}\nPred: {pred_label}")

plt.suptitle("Random Test Sample Predictions")
plt.show()