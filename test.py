import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from network import AConvnet
from dataset import VGG_SAMPLE_Dataset

if __name__ == '__main__':

    mu = 0.5
    gamma = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([mu], [gamma]),
                                    transforms.CenterCrop(88)])
    
    test_ds = VGG_SAMPLE_Dataset(transform = transform, train = False)
    test_dl = DataLoader(test_ds, batch_size = 4, shuffle = True)

    path2weight = './*.pt'

    weights = torch.load(path2weight)

    AConv = AConvnet().to(device)
    AConv.load_state_dict(weights)
    AConv.eval()

    correct = 0
    total = 0
    conf_matrix = np.zeros([10, 10])
    target_names = ('2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23')
    plt.figure(figsize=[6,6])
    with torch.no_grad():
        for data, label in test_dl:
            outputs = AConv(data.to(device)).detach().cpu()
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            _, label = torch.max(label, 1)
            correct += (predicted == label).sum().item()

            # Confusion Matrix
            for index in range(label.size(0)):
                conf_matrix[label[index], predicted[index]] += 1

    for i, a in enumerate(conf_matrix):
        for j, b in enumerate(a):
            plt.text(j, i, int(b), horizontalalignment="center", color='black')

    plt.imshow(conf_matrix, cmap='Blues')
    # plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.title('Acc : {:.2f}%, Data Num : {}'.format(100 * correct / total, total))
    plt.ylabel('Correct')
    plt.xlabel('Predicted')
    plt.show()
    plt.savefig('aa.png')

    print('[정확도 : {:.2f}, 데이터 개수 : {}]'.format(100 * correct / total, total))