# REFERENCES
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn 
import torch.optim as optim

class Trainer():
    def __init__(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


    def get_accuracy(self, loader, model):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct//total


    def train(self, model, lr, epochs=10, with_dropout=True):
        if with_dropout:
            model.train()
        else:
            model.eval()

        optimizer = optim.Adam(cnn_wdp.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        training_accuracies = []
        validation_accuracies = []

        # training loop
        for epoch in range(epochs):
            correct_train = 0
            total_train = 0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()

                # forward pass
                outputs = model(inputs)

                # loss calculation
                loss = criterion(outputs, labels)

                # backward pass
                loss.backward()
                optimizer.step()

                # find training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            # store training accuracy
            training_accuracy = 100 * correct_train//total_train
            training_accuracies.append(training_accuracy)

            # store validation accuracy
            validation_accuracy = self.get_accuracy(self.val_loader, model)
            validation_accuracies.append(validation_accuracy)

            print(f'Epoch {epoch +1}: training accuracy: {training_accuracy}%, validation accuracy {validation_accuracy}%')

        test_accuracy = self.get_accuracy(self.test_loader, model)
        print("---------------------------------------------------------")
        print('Finished Training')
        print(f"final training accuracy: {training_accuracies[-1]}%")
        print(f"final validation accuracy: {validation_accuracies[-1]}%")
        print(f"final test accuracy: {test_accuracy}%")

        return training_accuracies, validation_accuracies, test_accuracy
