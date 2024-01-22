# %%
import torch.nn as nn
import torchvision

model = torchvision.models.resnet101(pretrained = True)
print(model)
model.fc = nn.Linear(model.fc.in_features, 4)
print(model)