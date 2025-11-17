"""
Custom CNN Model for Deepfake Detection - PyTorch Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import json

class DeepfakeCNN(nn.Module):
    """
    Custom CNN architecture for deepfake detection
    5 convolutional blocks with batch normalization and dropout
    """
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        
        # Block 1: 32 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 2: 64 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 3: 128 filters
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 4: 256 filters
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 5: 512 filters
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class CNNModel:
    def __init__(self, model_path='models/cnn_trained.pth'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def build_model(self):
        """Build the CNN model"""
        self.model = DeepfakeCNN().to(self.device)
        return self.model
    
    def get_data_loaders(self, train_dir, val_dir, batch_size=32):
        """Create PyTorch data loaders"""
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # No augmentation for validation
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        return train_loader, val_loader
    
    def train(self, train_dir, val_dir, epochs=50, batch_size=32, learning_rate=0.001):
        """Train the model"""
        print("\n" + "="*70)
        print("TRAINING CUSTOM CNN MODEL")
        print("="*70)
        
        if self.model is None:
            self.build_model()
        
        # Get data loaders
        train_loader, val_loader = self.get_data_loaders(train_dir, val_dir, batch_size)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        print(f"\nTraining for {epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Device: {self.device}\n")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device).float().unsqueeze(1)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
                    
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{val_correct/val_total:.4f}'})
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{max_patience}")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                break
        
        print("\n" + "="*70)
        print("CNN MODEL TRAINING COMPLETE!")
        print("="*70)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {self.model_path}\n")
        
        return self.history
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, self.model_path)
    
    def load_model(self):
        """Load a trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        if self.model is None:
            self.build_model()
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.model.eval()
        print(f"Model loaded from {self.model_path}")
    
    def predict(self, image):
        """
        Predict if an image is real or fake
        Args:
            image: PIL Image or numpy array
        Returns:
            tuple: (prediction, confidence)
        """
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Convert to PIL if numpy array
        if not hasattr(image, 'convert'):
            from PIL import Image
            image = Image.fromarray(image)
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.sigmoid(output).item()
        
        # Model trained with ImageFolder: Fake=0, Real=1
        # Output close to 0 = Fake, close to 1 = Real
        # But we have single output neuron, so sigmoid > 0.5 means class 1 (Real)
        if prob > 0.5:
            prediction = "real"
            confidence = prob  # Confidence in Real
        else:
            prediction = "fake"
            confidence = 1 - prob  # Confidence in Fake
        
        return prediction, confidence
