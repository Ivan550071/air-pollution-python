import pickle
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from a6_ex1 import preprocess_data

def get_data_loaders(df: pd.DataFrame, batch_size: int = 32) -> tuple[DataLoader,
DataLoader, DataLoader]:
    """
    Prepares DataLoaders for training, validation, and testing datasets.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the cleaned air quality data with a datetime index.
    batch_size (int): Size of each batch for DataLoader.
    
    Returns:
    tuple: DataLoaders for training, validation, and testing datasets.
    """
    
    # here i do the following:
    # 1. filter numerical columns, which i use for training
    # 2. Scale the features using StandardScaler
    # 3. Convert DataFrame to PyTorch tensors (i drop PM2.5, because we will predict it
    feature_cols = [col for col in df.select_dtypes(include='number').columns if col != 'PM2.5']
    X = df[feature_cols].values
    y = df['PM2.5'].values.reshape(-1, 1)

    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scalar, f)# here i save the scaler for future use

    features = torch.tensor(X_scaled, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(features, targets)

    # Split dataset into train, validation, and test sets (80/10/10)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    zip_path = "PRSA2017_Data_20130301-20170228.zip"
    station = "Aotizhongxin"
    
    df = preprocess_data(zip_path, station)
    
    train_loader, val_loader, test_loader = get_data_loaders(df)
    
    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Validation Loader: {len(val_loader)} batches")
    print(f"Test Loader: {len(test_loader)} batches")