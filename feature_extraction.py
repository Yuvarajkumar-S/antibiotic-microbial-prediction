from Bio import SeqIO
from collections import Counter
import pandas as pd


def extract_kmers(sequence, k=4):

    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]

    return Counter(kmers)


def fasta_to_features(fasta_file):

    records = []
    ids = []

    for record in SeqIO.parse(fasta_file, "fasta"):

        seq = str(record.seq)

        kmer_counts = extract_kmers(seq)

        records.append(kmer_counts)

        ids.append(record.id)

    df = pd.DataFrame(records).fillna(0)

    df["sample"] = ids

    return df