import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization

# -----------------------------
# Parameters
# -----------------------------

MAX_LEN = 2000

mapping = {
"A":[1,0,0,0],
"T":[0,1,0,0],
"C":[0,0,1,0],
"G":[0,0,0,1]
}

# -----------------------------
# Encode DNA sequence
# -----------------------------

def encode_sequence(seq):

    encoded = []

    for base in seq[:MAX_LEN]:

        if base in mapping:
            encoded.append(mapping[base])
        else:
            encoded.append([0,0,0,0])

    while len(encoded) < MAX_LEN:
        encoded.append([0,0,0,0])

    return np.array(encoded)

# -----------------------------
# Load FASTA dataset
# -----------------------------

X = []
ids = []

print("Reading genomes...")

for record in SeqIO.parse("bacteria.10.2.genomic.fna","fasta"):

    seq = str(record.seq).upper()

    X.append(encode_sequence(seq))
    ids.append(record.id)

X = np.array(X)

# -----------------------------
# Load labels
# -----------------------------

labels = pd.read_csv("labels.csv")
labels = labels.set_index("sample")

y = []

for i in ids:

    if i in labels.index:
        y.append(labels.loc[i,"resistance"])
    else:
        y.append(0)

y = np.array(y)

# -----------------------------
# Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Improved CNN Model
# -----------------------------

inputs = Input(shape=(MAX_LEN,4))

x = Conv1D(128,7,activation="relu")(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

x = Conv1D(256,5,activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

x = Conv1D(256,3,activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

x = Flatten()(x)

x = Dense(256,activation="relu")(x)
x = Dropout(0.5)(x)

outputs = Dense(1,activation="sigmoid")(x)

model = Model(inputs,outputs)

model.compile(
optimizer="adam",
loss="binary_crossentropy",
metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train Model
# -----------------------------

history =model.fit(
    X_train,
    y_train,
    epochs=12,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight={0:2,1:1}
)

# -----------------------------
# Save Model
# -----------------------------

model.save("cnn_model.h5")

print("Model saved")

# -----------------------------
# Predictions
# -----------------------------

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# -----------------------------
# Metrics
# -----------------------------

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

print("\nAccuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1 Score:",f1)

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
cm,
annot=True,
fmt="d",
cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
