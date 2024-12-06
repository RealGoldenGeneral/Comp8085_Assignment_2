import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import sys

 # Create custom Dataset class
class YelpReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
            
    def __len__(self):
        return len(self.labels)
    
def predict_on_test(model, data_loader):
    batch_number = 0
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits

            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            batch_number += 1

            print(f"Batch {batch_number}/{len(data_loader)} done.")
            
    # Convert list of predictions and labels to numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_preds, all_labels

def evaluate(model, data_loader, device):
    batch_number = 0
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits

            preds.append(logits.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

            batch_number += 1

            print(f"Batch {batch_number}/{len(data_loader)} done.")
    preds = np.concatenate(preds, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Evaluate using mean squared error
    mse = mean_squared_error(true_labels, preds)
    return mse

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    batch_number = 0
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        logits = outputs.logits

        # Compute loss
        loss = loss_fn(logits, labels.float())
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_number}/{len(data_loader)} done.")

    return total_loss / len(data_loader)

def training_mode_bert(filename):
    chunk_number = 0
    # Read from the file name and create dataframes
    chunk_array = []
    try:
        print("Opening file...")
        with pd.read_json(filename, orient="records", lines=True, chunksize=40000) as reader:
            for chunk in reader:
                chunk_array.append(chunk)
                chunk_number += 1
                print(f"Chunk {chunk_number}/{len(reader)} done.")
    except Exception as e:
        print(f"An error occured with the file reader: {e}")
        return
    raw_dataset = chunk_array[0]

    print("Spliting dataset into test and validation files...")
    # Manually select portion of dataset
    train_size = 20000
    val_size = 30000

    # Manually slice dataset
    train_df = raw_dataset.iloc[:train_size]
    val_df = raw_dataset.iloc[train_size:val_size]

    print("Tokenizing data...")
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # Tokenize review texts
    train_texts = train_df['text'].tolist()
    val_texts = val_df['text'].tolist()

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    print("Extracting labels...")
    # Extract labels
    columns = ['stars', 'useful', 'cool', 'funny']

    train_labels = torch.tensor(train_df[columns].values)
    val_labels = torch.tensor(val_df[columns].values)

    # Create dataset objects for training and validation
    print("Creating datasets out of data...")
    train_dataset = YelpReviewDataset(train_encodings, train_labels)
    val_dataset = YelpReviewDataset(val_encodings, val_labels)

    # Create DataLoader objects
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=6)

    print("Creating model...")
    model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=4)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Define loss function (Mean Squared Error for regression)
    loss_fn = torch.nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run training and evaluation for multiple epochs
    print("Training model...")
    for epoch in range(2): # Adjust the number of epochs as necessary
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Epoch {epoch + 1} - Training loss: {train_loss}")

        mse = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1} - Validation mse: {mse}")
    
    with open('BERT.pkl', 'wb') as file:
        pickle.dump(model, file)

def inference_mode_bert(filename):
    chunk_number = 0
    chunk_array = []
    try:
        print("Opening model...")
        with open("./pickled_models/BERT.pkl", 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        print(f"An error has occured with loading the model: {e}")
        return
    
    try:
        print("Opening file...")
        with pd.read_json(filename, orient="records", lines=True, chunksize=40000) as reader:
            for chunk in reader:
                chunk_array.append(chunk)
                chunk_number += 1
                print(f"Chunk {chunk_number}/{len(reader)} done.")
    except Exception as e:
        print(f"An error occured with the file reader: {e}")
        return
    raw_dataset = pd.concat(chunk_array)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # Tokenize review texts
    print("Tokenizing data...")
    test_texts = raw_dataset['text'].tolist()

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    # Extract labels
    print("Extracting labels...")
    columns = ['stars', 'useful', 'cool', 'funny']

    test_labels = torch.tensor(raw_dataset[columns].values)

    # Create dataset object for testing
    print("Creating dataset out of data...")
    test_dataset = YelpReviewDataset(test_encodings, test_labels)

    # Create DataLoader object
    print("Creating data loader...")
    test_loader = DataLoader(test_dataset, batch_size=6)

    # Get predictions on the test set
    print("Running inference...")
    test_preds, test_labels = predict_on_test(model, test_loader)

    mse = mean_squared_error(test_labels, test_preds)
    print(f'Mean Squared Error on the test set: {mse}')

def main(model_type, mode, filename):
    model_type = model_type.lower()
    mode = mode.lower()
    if (mode == "training"):
        if (model_type == "transformer"):
            training_mode_bert(filename)
            return
        elif (model_type == "rnn"):
            # Insert functions for RNN here
            return
        elif (model_type == "naive bayes"):
            # Insert function for naive bayes here
            return
        else:
            print("Invalid arguments.")
    elif (mode == "inference"):
        if (model_type == "transformer"):
            inference_mode_bert(filename)
            return
        elif (model_type == "rnn"):
            # Insert functions for RNN here
            return
        elif (model_type == "naive bayes"):
            # Insert function for naive bayes here
            return
        else:
            print("Invalid arguments.")
    else:
        print("Invalid arguments.")

if __name__ == '__main__':
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Not enough arguments.")