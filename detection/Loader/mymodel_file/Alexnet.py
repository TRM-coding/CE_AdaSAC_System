from torch import nn
class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2,padding=0,dilation=1,ceil_mode=False)
        
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(6,6))

        self.flatten = nn.Flatten()

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        

        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        

        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool3(self.relu5(self.conv5(x)))

        x = self.avgpool(x)
        
        x = self.flatten(x)
        
        x = self.dropout1(self.relu6(self.fc1(x)))
        x = self.dropout2(self.relu7(self.fc2(x)))
        x = self.fc3(x)
        return x
