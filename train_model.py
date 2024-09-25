import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ProteinLSTM
from data_loader import load_sequences, preprocess_sequences, prepare_data, create_dataset

def train_model():
    # Load sequences
    sequences = load_sequences("data/uniprot_sprot.fasta", num_records=100)
    
    # Preprocess sequences
    encoded_sequences, aa_to_int, int_to_aa = preprocess_sequences(sequences)
    
    # Save the mappings
    torch.save(aa_to_int, 'models/aa_to_int.pth')
    torch.save(int_to_aa, 'models/int_to_aa.pth')
    
    # Prepare data
    sequence_length = 50
    X, y = prepare_data(encoded_sequences, sequence_length=sequence_length)
    
    # Create dataset and dataloader
    dataset = create_dataset(X, y)
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define the model
    vocab_size = len(aa_to_int) + 1
    embedding_dim = 50
    hidden_dim = 100
    num_layers = 2
    model = ProteinLSTM(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), 'models/protein_lstm_model.pth')

if __name__ == "__main__":
    train_model()
