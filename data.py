import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data(folder, batch_size):
    training_data = datasets.CIFAR10(root=folder, train=True, download=True,
		      transform=transforms.Compose([
			  transforms.ToTensor(),
			  transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
		      ]))

    validation_data = datasets.CIFAR10(root=folder, train=False, download=True,
		      transform=transforms.Compose([
			  transforms.ToTensor(),
			  transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
		      ]))

    train_loader = DataLoader(training_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 drop_last=True)

    valid_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   drop_last=True)

    data_var = np.var(training_data.data / 255.0)
    input_size = (3, 32, 32)

    return train_loader, valid_loader, data_var, input_size
