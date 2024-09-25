import torch
from model import ProteinLSTM
from data_loader import load_sequences, preprocess_sequences, prepare_data
import random

def generate_sequence(model, seed_seq, sequence_length, num_generated):
    model.eval()
    generated = seed_seq.copy()
    for _ in range(num_generated):
        input_seq = torch.tensor([generated[-sequence_length:]]).long()
        with torch.no_grad():
            output = model(input_seq)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            next_aa = torch.argmax(probabilities, dim=1).item()
        generated.append(next_aa)
    return generated

def main():
    # Load the mappings
    aa_to_int = torch.load('models/aa_to_int.pth')
    int_to_aa = torch.load('models/int_to_aa.pth')
    
    # Define the model parameters
    vocab_size = len(aa_to_int) + 1
    embedding_dim = 50
    hidden_dim = 100
    num_layers = 2
    model = ProteinLSTM(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    # Load the trained model
    model.load_state_dict(torch.load('models/protein_lstm_model.pth'))
    
    # Prepare a seed sequence
    sequences = load_sequences("data/uniprot_sprot.fasta", num_records=100)
    encoded_sequences, _, _ = preprocess_sequences(sequences)
    sequence_length = 50
    X, _ = prepare_data(encoded_sequences, sequence_length=sequence_length)
    X_tensor = torch.from_numpy(X).long()
    
    seed_idx = random.randint(0, len(X_tensor) - 1)
    seed_seq = X_tensor[seed_idx].tolist()
    
    # Generate sequence
    num_generated = 100
    generated_indices = generate_sequence(model, seed_seq, sequence_length, num_generated)
    generated_sequence = ''.join([int_to_aa.get(idx, '') for idx in generated_indices])
    print("Generated Protein Sequence:")
    print(generated_sequence)

if __name__ == "__main__":
    main()
