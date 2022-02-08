import torch.nn as nn

NUM_CLASSES = 2


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(11,10, kernel_size=3, stride=2,padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(kernel_size=2), 

            nn.Conv2d(10,20, kernel_size=3, stride=1,padding=1),  
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(20, 40, kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(40, 40, kernel_size=3,stride=1,padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(40, 20, kernel_size=3, stride=1,padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.classifier = nn.Sequential(   
            nn.Dropout(),               
            nn.Linear(500, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
