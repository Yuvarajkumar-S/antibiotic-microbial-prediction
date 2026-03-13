import numpy as np
from Bio import SeqIO
from tensorflow.keras.models import load_model

# Maximum genome length used in training
MAX_LEN = 2000

# DNA one-hot encoding
mapping = {
"A":[1,0,0,0],
"T":[0,1,0,0],
"C":[0,0,1,0],
"G":[0,0,0,1]
}

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


print("Loading CNN model...")
model = load_model("cnn_model.h5")


print("Predicting resistance...\n")

for record in SeqIO.parse("bacteria.10.2.genomic.fna","fasta"):

    seq = encode_sequence(str(record.seq).upper())

    seq = np.expand_dims(seq,axis=0)

    pred = model.predict(seq)

    confidence = float(pred[0][0])

    if confidence > 0.7:
        prediction = "Resistant"
    else:
        prediction = "Susceptible"

    print(record.id, "→", label, "(confidence:", round(confidence,3),")")
