import torch

# ---------------------------------------------------------------------------- #
#                                PREPARE DATASET                               #
# ---------------------------------------------------------------------------- #

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = MNIST(root='/code/data/MNIST', download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# ---------------------------------------------------------------------------- #
#                            TRAINING AND EVALUATION                           #
# ---------------------------------------------------------------------------- #

from tqdm import tqdm

def train(model, optmz, crtrn, dl, **args):
    model.train()
    
    loss = 0
    for images, labels in tqdm(dl):
        images = images.to(device)
        labels = labels.to(device)

#         where does nan come from
#         torch.autograd.set_detect_anomaly(True)
        output = model.forward(images, **args).squeeze() 
        batch_loss = crtrn(output, labels)
        
        optmz.zero_grad()
#         where does nan come from
#         with torch.autograd.detect_anomaly():
        batch_loss.backward()
        optmz.step()
        
        loss += batch_loss.item()
    
    return loss

@torch.no_grad()
def eval(model, dl, **args):
    model.eval()

    total, n_correct = 0, 0
    for images, labels in tqdm(dl):
        images = images.to(device)
        labels = labels.to(device)

        output = model.forward(images, **args).squeeze()
        predic = torch.max(output, dim=1)
        correct = predic.indices == labels

        total += labels.size(0)
        n_correct += correct.sum().item()

    return n_correct / total

# ---------------------------------------------------------------------------- #
#                                     MODEL                                    #
# ---------------------------------------------------------------------------- #

import sys
import torch.nn as nn
import Lipschitz.nn as Ln

class MLP(nn.Module):
    def __init__(self, *layers, p=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.p = p
    
    def forward(self, inputs, p=None):
        for layer in self.layers:
            try:
                inputs = layer(inputs, self.p if p is None else p)
            except Exception:
                inputs = layer(inputs)
        return inputs

### A simple FC network
net = MLP(

    nn.Flatten(),
    Ln.Linear(28*28, 256),
    Ln.Linear(256, 64),
    Ln.Linear(64, 10),

p=float(sys.argv[1])).to(device)

# ---------------------------------------------------------------------------- #
#                                      RUN                                     #
# ---------------------------------------------------------------------------- #

optmz = torch.optim.Adam(net.parameters(), 1e-3)
crtrn = torch.nn.CrossEntropyLoss()

loss_list, accu_list = [], []
for epoch in range(32):
    loss = train(net, optmz, crtrn, dataloader)
    accu = eval(net, dataloader)

    print(f'[{sys.argv[1]}] Epoch {epoch+1} average loss: {loss}, accuracy: {accu}')
    loss_list.append('{:12.6f}'.format(loss))
    accu_list.append('{:12.6f}'.format(accu))

with open(f'mnist.{sys.argv[1]}.txt', 'w') as output:
    output.write(', '.join(loss_list) + '\n')
    output.write(', '.join(accu_list) + '\n')
