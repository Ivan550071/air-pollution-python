import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from a6_ex4 import PM_Model
from a6_ex1 import preprocess_data
from a6_ex3 import get_data_loaders

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-2)
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save model parameters
    torch.save(model.state_dict(), "model.pt")

    # Test step and plot predictions
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            y_true.extend(targets.cpu().numpy().flatten())
            y_pred.extend(outputs.cpu().numpy().flatten())

    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True PM2.5", alpha=0.7)
    plt.plot(y_pred, label="Predicted PM2.5", alpha=0.7)
    plt.title("True vs Predicted PM2.5 on Test Set")
    plt.xlabel("Sample")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model_prediction.pdf")

if __name__ == "__main__":
    zip_path = "PRSA2017_Data_20130301-20170228.zip"
    station = "Aotizhongxin"

    df = preprocess_data(zip_path, station)
    train_loader, val_loader, test_loader = get_data_loaders(df)

    input_size = next(iter(train_loader))[0].shape[-1] #-1 cuz PM2.5 is the target of training
    model = PM_Model(input_size=input_size)

    train_model(model, train_loader, val_loader, test_loader)