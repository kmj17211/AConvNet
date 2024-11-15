import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import time
import matplotlib.pyplot as plt

from network import AConvnet
from dataset import VGG_SAMPLE_Dataset

def saveModel(state_dict):
    path = './*.pt'
    torch.save(state_dict, path)

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mu = 0.5
    gamma = 0.5
    lr = 1e-4
    num_epoch = int(100) 
    batch_size = 32

    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize([mu], [gamma]),
                                    transforms.CenterCrop(88)])
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                  transforms.CenterCrop(88)])

    train_ds = VGG_SAMPLE_Dataset(transform = transform, train = True)
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    
    AConv = AConvnet().to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(AConv.parameters(), lr = lr)
    sche_opt = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)

    AConv.train()
    start_time = time.time()

    img, _ = train_ds[20]
    plt.imshow(img.squeeze())
    plt.colorbar()
    plt.savefig('a.png')

    print('Start Train')
    for epoch in range(num_epoch):
        for data, label in train_dl:
            
            data, label = data.to(device), label.to(device)
            
            AConv.zero_grad()

            outputs = AConv(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
        
        print('Epoch: {}, Train Loss: {:.3}, Time: {:.2}min'.format(epoch, loss, (time.time()-start_time)/60))
        sche_opt.step()
    
    print('Finish')

    saveModel(AConv.state_dict())