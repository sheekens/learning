from torchvision.models import resnet18
from torch import nn
from torch.functional import Tensor
from varname.helpers import debug

class ResnetModel(nn.Module):
    def __init__(
            self,
            num_classes
            ):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = resnet18()
        self.classification_head = nn.Linear(
            in_features=128,
            out_features=self.num_classes
            )
        self.softmax = nn.Softmax(1)

    def forward(self, x: Tensor): 
        x = self.resnet(x)
        x = self.classification_head(x)
        return x
        