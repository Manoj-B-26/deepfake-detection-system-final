"""
ResNet-152 Model for Deepfake Detection - PyTorch Implementation
Using transfer learning with pre-trained ResNet-152
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
from tqdm import tqdm

class ResNetModel:
    def __init__(self, model_path='models/resnet_trained.pth'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def build_model(self, freeze_base=True):
        """Build ResNet-152 model with custom classification head"""
        # Load pre-trained ResNet-152
        base_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        
        # Freeze base model if specified
        if freeze_base:
            for param in base_model.parameters():
                param.requires_grad = False
        
        # Replace final layer with custom classifier
        num_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self.model = base_model.to(self.device)
        return self.model
    
    def unfreeze_base_model(self, num_layers=50):
        """Unfreeze last N layers of base model for fine-tuning"""
        # Get all parameters
        all_params = list(self.model.parameters())
        
        # Unfreeze last num_layers
        for param in all_params[-num_layers:]:
            param.requires_grad = True
        
        print(f"Unfroze last {num_layers} layers for fine-tuning")
    
    def get_data_loaders(self, train_dir, val_dir, batch_size=16):
        """Create PyTorch data loaders"""
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    
    def train(self, train_dir, val_dir, epochs=30, batch_size=16, learning_rate=0.0001):
        """Train the model"""
        print("\n" + "="*70)
        print("TRAINING RESNET-152 MODEL")
        print("="*70)
        
        if self.model is None:
            self.build_model(freeze_base=True)
        
        # Get data loaders
        train_loader, val_loader = self.get_data_loaders(train_dir, val_dir, batch_size)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
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
        print("RESNET-152 MODEL TRAINING COMPLETE!")
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
            self.build_model(freeze_base=False)
        
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
            confidence = torch.sigmoid(output).item()
        
        prediction = "fake" if confidence > 0.5 else "real"
        
        return prediction, confidence
