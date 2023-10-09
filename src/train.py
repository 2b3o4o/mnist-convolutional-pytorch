import logging
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class Trainer:
    def __init__(self, model):
        self.device = torch.device("cuda")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        self.test_data = test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        train_split = 0.8
        train_size = int(train_split * len(train_data))
        val_size = len(train_data) - train_size
        train_subset, val_subset = random_split(train_data, [train_size, val_size])
        self.train_loader = DataLoader(dataset=train_subset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(dataset=val_subset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def train_one_epoch(self):
        self.model.train(True)
        batches = 0
        avg_loss = 0
        for step, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(features)
            loss = self.loss_func(preds, labels)
            loss.backward()
            self.optimizer.step()
            
            avg_loss += loss
            batches = step
        
        avg_loss = avg_loss / batches
        print(f"Average loss for training batches in this epoch: {avg_loss}")

        self.model.train(False)
        batches = 0
        avg_loss = 0
        for step, (features, labels) in enumerate(self.val_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(features)
            loss = self.loss_func(preds, labels)
            
            avg_loss += loss
            batches = step

        avg_loss = avg_loss / batches
        print(f"Average loss for validation batches in this epoch: {avg_loss}")

    def train(self, epochs):
        self.model.train(True)
        for i in range (0, epochs):
            print(f"Beginning epoch {i}...")
            self.train_one_epoch()
            print("")

    def evaluate_accuracy(self):
        self.model.train(False)
        preds = []
        for features, labels in self.test_loader:
            with torch.no_grad():
                batch_preds = self.model(features)
                preds.extend(batch_preds.tolist())
        preds_tensor = torch.tensor(preds)
        category_preds = torch.argmax(preds_tensor, dim=1)

        actual = self.test_data.targets
        assert len(actual) == len(preds)

        logging.info(f"Accuracy: {sum([int(actual[i] == category_preds[i]) for i in range(0, len(actual))]) / len(actual)}")