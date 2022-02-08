import torch.nn as nn

NUM_CLASSES = 2


class LeNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(LeNet, self).__init__()
        
        self.features = nn.Sequential(
           
            nn.Conv2d(11,10, kernel_size=5, stride=1,padding=0), 
            nn.ReLU(inplace=True),  
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(10,20, kernel_size=5, stride=1,padding=0),  
            nn.ReLU(inplace=True), 
            nn.AvgPool2d(kernel_size=2),    
            )
        

        self.classifier = nn.Sequential(   
            nn.Dropout(),               
            nn.Linear(5780, 120),
            nn.ReLU(inplace=True), 
            
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),  
            
            nn.Linear(84, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
