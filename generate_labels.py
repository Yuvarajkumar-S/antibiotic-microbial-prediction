from Bio import SeqIO
import pandas as pd
import random

samples = []

for record in SeqIO.parse("bacteria.10.2.genomic.fna", "fasta"):

    samples.append({
        "sample": record.id,
        "resistance": random.randint(0,1)
    })

df = pd.DataFrame(samples)

df.to_csv("labels.csv", index=False)

print("labels.csv generated successfully")