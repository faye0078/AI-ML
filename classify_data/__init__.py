import torchvision
import torch
from classify_data.mnist import mnist
def make_dataloaders(dataset_name):
    if dataset_name == 'mnist':
        train_set, test_set = mnist()
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)
        return train_loader, test_loader