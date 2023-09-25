# WISET
## 2022 WISET - CapsNet

### 1. Capsnet_Conv3
- add Conv2D layer in original Capsnet to increase accuracy
- Batch size = 128, Epoch = 10

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

### 2. Res-CapsNet
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

