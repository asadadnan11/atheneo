import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification

class TeamIdentifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, texts, teams, epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Train the team identifier model
        
        Args:
            texts (list): List of text samples
            teams (list): List of team labels (0 or 1)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
        """
        try:
            # Convert texts to input tensors
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            labels = torch.tensor(teams)
            
            # Create DataLoader
            dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Setup optimizer
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in dataloader:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
            
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise 