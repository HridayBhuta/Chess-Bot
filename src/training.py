import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from model import ChessStyleBot

class ChessDataset(Dataset):
    def __init__(self, pt_file):
        print(f"Loading data from {pt_file}...")
        data = torch.load(pt_file)
        self.X = data['positions']
        self.y = data['moves']
        print(f"Dataset loaded: {len(self.X)} positions found (Augmented).")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model():
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = "my_style_bot.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    full_dataset = ChessDataset("data/training_data.pt")
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ChessStyleBot().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    print(f"Starting training on {train_size} positions...")
    
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss = criterion(output, target)
                total_val_loss += val_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"\n>>> Epoch {epoch+1} Results:")
        print(f"    Avg Train Loss: {avg_train_loss:.4f}")
        print(f"    Avg Val Loss:   {avg_val_loss:.4f}")
        print(f"    Learning Rate:  {optimizer.param_groups[0]['lr']}")
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"    *** New Best Model Saved! ***\n")
        else:
            print("\n")

    print(f"Training Complete! Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_model()