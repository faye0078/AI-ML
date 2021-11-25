from torchvision import datasets, transforms

def mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.1307),std=(0.3081))])
    data_train = datasets.MNIST(root = "../data/MNIST",
                                transform=transform,
                                train = True,
                                download = True)

    data_test = datasets.MNIST(root="../data/MNIST",
                               transform = transform,
                               train = False)
    return data_train, data_test