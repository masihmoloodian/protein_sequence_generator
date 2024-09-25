import numpy as np
from Bio import SeqIO
import torch
from torch.utils.data import TensorDataset

def load_sequences(filename, num_records=100):
    sequences = []
    for i, record in enumerate(SeqIO.parse(filename, "fasta")):
        if i >= num_records:
            break
        sequences.append(str(record.seq))
    return sequences

def preprocess_sequences(sequences):
    amino_acids = set(''.join(sequences))
    aa_to_int = {aa: i+1 for i, aa in enumerate(sorted(amino_acids))}
    int_to_aa = {i+1: aa for aa, i in aa_to_int.items()}
    encoded_sequences = [[aa_to_int[aa] for aa in seq] for seq in sequences]
    return encoded_sequences, aa_to_int, int_to_aa

def prepare_data(encoded_sequences, sequence_length=50):
    X = []
    y = []
    for seq in encoded_sequences:
        for i in range(0, len(seq) - sequence_length):
            seq_in = seq[i:i + sequence_length]
            seq_out = seq[i + sequence_length]
            X.append(seq_in)
            y.append(seq_out)
    X = np.array(X)
    y = np.array(y)
    return X, y

def create_dataset(X, y):
    X_tensor = torch.from_numpy(X).long()
    y_tensor = torch.from_numpy(y).long()
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset
