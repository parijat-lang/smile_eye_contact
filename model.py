import torch
import torch.nn as nn
import torchvision.models as models


class CustomVGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomVGG16, self).__init__()
        # Load the VGG16 model
        self.vgg16 = models.vgg16(pretrained=True).features
        # Freeze the layers
        for param in self.vgg16.parameters():
            param.requires_grad = True
        
        # Replace the classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.vgg16(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CustomVGG16().to(device)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

