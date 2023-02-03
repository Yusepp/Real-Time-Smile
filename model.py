from time import time

import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score

from early import EarlyStopper
from log import *


class SmileDetector(nn.Module):
    def __init__(self, net='MNet-L', freeze=False):
        super(SmileDetector, self).__init__()
        
        self.loss_fn = None
        self.opt = None
        self.early_stopping = None
        self.freeze = freeze
        self.net = net
        
        self.backbone, backbone_features = self.get_backbone()
        
        # Landmark Features
        self.flatten = nn.Flatten()
        
        # MLP Blocks
        self.fc1 = nn.Linear(backbone_features, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.drop1 = nn.Dropout(p=0.25)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.drop2 = nn.Dropout(p=0.25)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.drop3 = nn.Dropout(p=0.25)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.drop4 = nn.Dropout(p=0.25)
        
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(num_features=64)
        self.drop5 = nn.Dropout(p=0.25)
        
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(num_features=32)
        
        self.fc7 = nn.Linear(32, 1)
    
    def forward(self, images):
        # Extract backbone features
        image_features = self.backbone(images)
        x = image_features.view(image_features.size(0), -1)
        
        # Feed the MLP Blocks   
        # FC -> ReLU -> BatchNorm -> Dropout -> Next Layer
        x = self.drop1(self.bn1(F.relu(self.fc1(x))))
        x = self.drop2(self.bn2(F.relu(self.fc2(x))))
        x = self.drop3(self.bn3(F.relu(self.fc3(x))))
        x = self.drop4(self.bn4(F.relu(self.fc4(x))))
        x = self.drop5(self.bn5(F.relu(self.fc5(x))))
        x = self.bn6(F.relu(self.fc6(x)))
        
        # Binary Classfication
        x = torch.sigmoid(self.fc7(x))
        
        return x

    
    def fit(self, train_data, batch_size=1, epochs=3, val_data=None, patience=0, logging=True):
        # Detect GPU or CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Start logging wandb
        if logging:
            start_log(self)
        
        # Metrics
        train_acc  = BinaryAccuracy().to(device)
        train_prec  = BinaryPrecision().to(device)
        train_rec  = BinaryRecall().to(device)
        train_f1 = BinaryF1Score().to(device)
        
        # Optimizer and loss function
        self.loss_fn = nn.BCELoss()
        self.opt = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        steps = len(train_data.dataset)//batch_size
        
        # Setting up EarlyStopping
        if patience > 0:
            self.early_stopping = EarlyStopper(patience=patience)
        val_loss = None
        stop = None

        # Train the model
        for epoch in range(epochs):
            torch.cuda.synchronize()
            start_epoch = time()
            running_loss = 0.0
            self.train()
            
            for i, data in enumerate(train_data):
                # Load Batch
                imgs, labels = data
                imgs, labels = imgs.to(device).to(torch.float32), labels.to(device).unsqueeze(1).to(torch.float32)
                
                # Gradient params to zero
                self.opt.zero_grad()
                
                # Forward
                outputs = self.forward(imgs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.opt.step()
                
                # Stats
                running_loss += loss.item()
                acc, prec, rec, f1 = train_acc(outputs, labels), train_prec(outputs, labels), train_rec(outputs, labels), train_f1(outputs, labels)

                # Print Batch
                print(f"Epoch {epoch+1:02d}/{epochs:02d} Batch {i:02d}/{steps:02d} --- loss: {running_loss/(i+1):.4f} --- accuracy: {acc:.4f} --- precision: {prec:.4f} --- recall: {rec:.4f} --- f1: {f1:.4f}", end="\r")
            
            torch.cuda.synchronize()            
            end_epoch = time()
            
            # Print Final Epoch Metrics
            acc, prec, rec, f1 = train_acc.compute(), train_prec.compute(), train_rec.compute(), train_f1.compute()
            print(f"Epoch {epoch+1:02d}/{epochs:02d} Batch {steps:02d}/{steps:02d} --- loss: {running_loss/steps:.4f} --- accuracy: {acc:.4f} --- precision: {prec:.4f} --- recall: {rec:.4f} --- f1: {f1:.4f}", end=" ")
            print(f"--- epoch_time: {end_epoch-start_epoch:.2f}s --- time/step: {(end_epoch-start_epoch)/steps:.2f}s")
            
            # Log Epoch Wandb
            if logging:
                log_epoch({"Train Loss": running_loss/steps, "Train Accuracy": acc, "Train Precision": prec, "Train Recall": rec, "Train F1": f1})
            
            # Run Validation step
            if val_data:
                val_loss = self.evaluate(val_data, batch_size)
                
            # Check Early stopping
            if val_loss and patience > 0:
                stop = self.early_stopping.early_stop(loss=val_loss, model_weights=self.state_dict(), epoch=epoch+1)
            elif not val_loss and patience > 0:
                stop = self.early_stopping.early_stop(loss=running_loss/steps, model_weights=self.state_dict(), epoch=epoch+1)
            
            if stop:
                if self.early_stopping.best_model_weights:
                    print(f"Early stopping! Restoring weights from best epoch {self.early_stopping.best_epoch} with loss {self.early_stopping.min_loss}")
                    self.load_state_dict(self.early_stopping.best_model_weights)
                del self.early_stopping
                break
            print("")
            
        print("Finished training!")

    
    def evaluate(self, data, batch_size = 1, logging=True):
        # Detect GPU or CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set eval
        self.eval()
        running_loss = 0.0
        steps = len(data.dataset)//batch_size
        
        # Funcs
        if not self.loss_fn:
            self.loss_fn = nn.BCELoss()
            self.opt = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        
        # Metrics
        val_acc  = BinaryAccuracy().to(device)
        val_prec  = BinaryPrecision().to(device)
        val_rec  = BinaryRecall().to(device)
        val_f1 = BinaryF1Score().to(device)
        
        # Run inference
        with torch.no_grad():
            torch.cuda.synchronize()            
            start_epoch = time()
            for i, data in enumerate(data):
                # Load Batch
                imgs, labels = data
                imgs, labels = imgs.to(device).to(torch.float32), labels.to(device).to(torch.float32)
                # Output Data
                outputs = self.forward(imgs).unsqueeze(1)
                outputs, labels = torch.reshape(outputs, (-1,)), torch.reshape(labels, (-1,))
                running_loss += self.loss_fn(outputs, labels)
                
                # Print Metrics per sample
                acc, prec, rec, f1 = val_acc(outputs, labels), val_prec(outputs, labels), val_rec(outputs, labels), val_f1(outputs, labels)
                print(f"Val. loss: {running_loss/(i+1):.4f} --- Val. accuracy: {acc:.4f} --- Val. precision: {prec:.4f} --- Val. recall: {rec:.4f} --- Val. f1: {f1:.4f}", end="\r")
           
            torch.cuda.synchronize()            
            end_epoch = time()
            
            # Print Final Evaluation Metrics
            acc, prec, rec, f1 = val_acc.compute(), val_prec.compute(), val_rec.compute(), val_f1.compute()
            print(f"Val. loss: {running_loss/steps:.4f} --- Val. accuracy: {acc:.4f} --- Val. precision: {prec:.4f} --- Val. recall: {rec:.4f} --- Val. f1: {f1:.4f}", end=" ")
            print(f"--- inference_time: {end_epoch-start_epoch:.2f}s --- time/step: {(end_epoch-start_epoch)/steps:.2f}s")
            
            # Log Validation Wandb
            if logging:
                log_epoch({"Val. Loss": running_loss/steps, "Val. Accuracy": acc, "Val. Precision": prec, "Val. Recall": rec, "Val. F1": f1})
            
        return running_loss/steps
    
    
    
    
    
    def get_backbone(self):
        
        if self.net == 'MNet-L':
            # Load MNetV3_Large pretrained on  ImageNet
            backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            
            # If freeze, we freeze all except last 2 layers
            if self.freeze:
                for param in list(self.backbone.parameters())[:-2]:
                    param.requires_grad = False

            # We build the backbone
            backbone_features = 960 # Number of Features obtained by the backbone
            modules = list(backbone.children())[:-1]
            backbone = nn.Sequential(*modules)
        
        elif self.net == 'MNet-S':
            # Load MNetV3_Small pretrained on  ImageNet
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            
            # If freeze, we freeze all except last 2 layers
            if self.freeze:
                for param in list(self.backbone.parameters())[:-2]:
                    param.requires_grad = False

            # We build the backbone
            backbone_features = 576 # Number of Features obtained by the backbone
            modules = list(backbone.children())[:-1]
            backbone = nn.Sequential(*modules)
        
        
        elif self.net == 'RNet-50':
            # Load ResNet-50 pretrained on  ImageNet
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            
            # If freeze, we freeze all except last 2 layers
            if self.freeze:
                for param in list(backbone.parameters())[:-2]:
                    param.requires_grad = False

            # We build the backbone
            backbone_features = 2048 # Number of Features obtained by the backbone
            modules = list(backbone.children())[:-1]
            backbone = nn.Sequential(*modules)
        
        
        elif self.net == 'ShNet':
            # Load ShuffleNetV2 pretrained on  ImageNet
            backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
            
            # If freeze, we freeze all except last 2 layers
            if self.freeze:
                for param in list(self.backbone.parameters())[:-2]:
                    param.requires_grad = False

            # We build the backbone
            backbone_features = 1024 # Number of Features obtained by the backbone
            modules = list(backbone.children())[:-1] + [nn.AvgPool2d(7)]
            backbone = nn.Sequential(*modules)
            
        
        
        return backbone, backbone_features
