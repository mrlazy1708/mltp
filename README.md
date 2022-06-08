# mltp: Machine Learning T[heory|erm] Project
```
This repo contains code of our term project for Machine Learning course in Peking University.

File structure: 2 directories, 14 files
├── Lipschitz                    : Lp-Network components
│   ├── core.py                  :   Functions used to calculate Lp-Neuron output
│   └── nn.py                    :   Lp-Neuron implementation of some torch.nn layers
│
├── mnist.py                     : Experiment on MNIST to investigate expressive power
├── robust.py                    : Experiment on MNIST to compare certified robustness
├── plot.ipynb                   : Visualization scripts
│
└── output                       : Output of our experiments
    ├── ...

Reproduce results:
- python mnist.py <p>
- python robust.py
```
