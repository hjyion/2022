# WISET
## 2022 WISET - CapsNet

### About Capsule Network
- CapsNet
##### - Related Work
- NASCaps : Automated framework for hardware-aware NAS that covers traditional DNN and CapsNet
  
   Based on genetic NSGA-II algorithm, uses configurations of underlying HW accelerator and given dataset for training as well as collections of possible types of layers as input. Creating layer library that includes conv layer, capsule layer, CapsCell and FlatCaps. Automated search in NASCaps starts with N randomly generated DNNs, evaluating accuracy after trained with limited epoch and finally pareto-optimal DNN solutions are fully trained.

  NASCaps is proper framework when we have limited design time and training source + when DNN design needs short training duration -> make CapsNet-based DNN's deployment more easier in resource-constrained IoT/Edge device

#### 1. Capsnet_Conv3
- add Conv2D layer in original Capsnet to increase accuracy
- Batch size = 128, Epoch = 10
- Final Accuracy : 71.1%

```python
        # Conv2d layer1
        self.conv1 = nn.Conv2d(1, 32, 9)
        self.relu1 = nn.ReLU(inplace=True)

        # Batch Normalization1
        self.bn1 = nn.BatchNorm2d(32)

        # Conv2d layer2
        self.conv2 = nn.Conv2d(32, 64, 9)
        self.sigmoid2 = nn.Sigmoid()

        # Batch Normalization2
        self.bn2 = nn.BatchNorm2d(64)

        # Conv2d layer3
        self.conv3 = nn.Conv2d(64, 256, 9)
        self.relu3 = nn.ReLU(inplace=True)
```

#### 2. Res-CapsNet
- add Residual Block in CapsNet
- Batch size = 128, Epoch = 10
- Accuracy : 57.2%

```python
class Residual_Block(nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim):
      super(Residual_Block,self).__init__()
        # Residual Block
      self.residual_block = nn.Sequential(
                nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1),
                nn.ReLU,
                nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1),
            )            
      self.relu = nn.ReLU()
                  
    def forward(self, x):
       out = self. residual_block(x)  # F(x)
       out = out + x  # F(x) + x
       out = self.relu(out)
       return out
```

