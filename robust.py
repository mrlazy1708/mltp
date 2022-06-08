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
def eval(model, epsln, dl, **args):
    model.eval()

    total, n_correct, n_certify = 0, 0, 0
    for images, labels in tqdm(dl):
        images = images.to(device)
        labels = labels.to(device)

        output = model.forward(images, **args).squeeze()
        output = model(images).squeeze()
        predic = torch.max(output, dim=1)
        second = torch.kthvalue(output, 9, dim=1)
        correct = predic.indices == labels
        certify = correct.logical_and(predic.values - second.values > epsln * 2)

        total += labels.size(0)
        n_correct += correct.sum().item()
        n_certify += certify.sum().item()

    return n_correct / total, n_certify / total

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
    Ln.Linear(28*28, 2048),
    Ln.Linear(2048, 10),

p=float(sys.argv[1])).to(device)

# ---------------------------------------------------------------------------- #
#                                      RUN                                     #
# ---------------------------------------------------------------------------- #

epsln = 8/255
optmz = torch.optim.Adam(net.parameters(), 1e-3)
crtrn = torch.nn.CrossEntropyLoss()

loss_list, accu_list, cert_list = [], [], []
for epoch in range(20):
    loss = train(net, optmz, crtrn, dataloader)
    accu, cert = eval(net, epsln, dataloader)

    print(f'Epoch {epoch+1} average loss: {loss}, accuracy(certified): {accu}({cert})')
    loss_list.append('{:12.6f}'.format(loss))
    accu_list.append('{:12.6f}'.format(accu))
    cert_list.append('{:12.6f}'.format(cert))

with open(f'mnist.robust.txt', 'w') as output:
    output.write(', '.join(loss_list) + '\n')
    output.write(', '.join(accu_list) + '\n')
    output.write(', '.join(cert_list) + '\n')
