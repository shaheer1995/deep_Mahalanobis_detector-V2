from torchvision import models
from torch import nn

def bind_method(obj, method):
    """Bind new methods to already instantiated objects."""
    setattr(obj, method.__name__, method.__get__(obj))


def create_model(num_classes=10):
    model = models.vgg16(pretrained=True)
        
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # Re-implement final classification layer
    model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=num_classes, bias=True)

    return model


def penultimate_forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    
    # Break up self.classifier
    penultimate = self.classifier[:5](x)
    out = self.classifier[5:](penultimate)
    return out, penultimate


# function to extact the multiple features
def feature_list(self, x):
    # import pdb; pdb.set_trace()
    out_list = []
    x = self.features(x)    
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)

    for l in self.classifier:
        x = l(x)
        out_list.append(x)
    
    return x, out_list


def intermediate_forward(self, x, layer_index):
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    
    # Break up self.classifier
    out = self.classifier[:layer_index+1](x)
    return out


def VGG16(num_classes):
    model = create_model(num_classes)
    bind_method(model, penultimate_forward)
    bind_method(model, feature_list)
    bind_method(model, intermediate_forward)

    return model

