import torchvision.models as models

resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
print(resnext50_32x4d)