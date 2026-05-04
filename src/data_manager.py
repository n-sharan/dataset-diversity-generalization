from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

class DataManager():
    def __init__(self, batch_size):
        self.b = batch_size
        self.transform = {
            "train" : transforms.Compose(
                [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val" : transforms.Compose(
                [transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "test" : transforms.Compose(
                [transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    def get_loader(self, path):
        dataset = datasets.ImageFolder(root=path, transform=None)

        # split proportions
        train_size = int(0.6 * len(dataset))  # 60% for training
        val_size = int(0.1 * len(dataset))    # 10% for validation
        test_size = len(dataset) - train_size - val_size  # 30 for testing

        # split the dataset into train, validation, and test sets
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # apply transformations
        train_dataset.dataset.transform = self.transform['train']  # assign training transform
        val_dataset.dataset.transform = self.transform['val']  # assign validation transform
        test_dataset.dataset.transform = self.transform['test']  # assign test transform

        # dataloaders
        train_loader = DataLoader(DBI_train_dataset, batch_size=b, shuffle=True)
        val_loader = DataLoader(DBI_val_dataset, batch_size=b, shuffle=False)
        test_loader = DataLoader(DBI_test_dataset, batch_size=b, shuffle=False)
    
        return train_loader, val_loader, test_loader
