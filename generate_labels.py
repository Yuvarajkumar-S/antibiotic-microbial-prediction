from Bio import SeqIO
import pandas as pd

samples = []
labels = []

for record in SeqIO.parse("bacteria.10.2.genomic.fna", "fasta"):
    seq = str(record.seq)

    # count GC content
    gc_count = seq.count("G") + seq.count("C")
    gc_ratio = gc_count / len(seq)

    # create labels based on GC content
    if gc_ratio > 0.5:
        label = 1
    else:
        label = 0

    samples.append(record.id)
    labels.append(label)

df = pd.DataFrame({
    "sample": samples,
    "resistance": labels
})

df.to_csv("labels.csv", index=False)

print("labels.csv created successfully")
print(df["resistance"].value_counts())
