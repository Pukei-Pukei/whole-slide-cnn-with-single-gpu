from torch import nn
import timm


def Extractor(pretrained=True):
    return timm.create_model('dm_nfnet_f0', pretrained=pretrained, num_classes=0)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(3072, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

