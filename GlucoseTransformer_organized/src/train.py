import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=1e-3):
    """
    Train the transformer model with early stopping.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        
    Returns:
        Tuple of (training_losses, validation_losses)
    """
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    best_model = None
    
    train_losses = []
    val_losses = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        val_loss = evaluate_model(model, val_loader, criterion)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict().copy()
        else:
            patience_counter += 1
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return train_losses, val_losses


def evaluate_model(model, data_loader, criterion):
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing evaluation data
        criterion: Loss function
        
    Returns:
        Average loss on the dataset
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(data_loader)
    return val_loss