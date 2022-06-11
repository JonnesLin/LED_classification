import torch
import torchvision
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer
from model.dlb_model import DLBModel
from config import Config
from dataset.transform import train_transform, test_transform
from torchvision import transforms, datasets
import torchvision.models as models
import timm

def imageset_loader():
    num_label = 4
    kwargs = {'num_workers': 8, 'pin_memory': True}
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_train = transforms.Compose([transforms.RandomResizedCrop((224, 224)), 
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(80),
                                          transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5),
                                          transforms.ToTensor(), 
                                          normalize, ])
    transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(), normalize, ])
    
    trainset = datasets.ImageFolder(root='../FRSKD/data/led/train/', transform=transform_train)
    testset = datasets.ImageFolder(root='../FRSKD/data/led/valid/', transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


seed_everything(Config.seed)

backbone = torchvision.models.resnet50(pretrained=True, progress=True)
# backbone = models.efficientnet_b2(pretrained=True)
# backbone = timm.create_model('efficientnet_b0', pretrained=True)
dlb_model = DLBModel(Config, backbone)

trainer = Trainer(gpus=[0], max_epochs=Config.epochs)


def run():
    # torch.multiprocessing.freeze_support()
    train_loader, test_loader = imageset_loader()

    trainer.fit(dlb_model, train_loader, test_loader)
    print('loop')

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    train_loader, test_loader = imageset_loader()
    trainer.fit(dlb_model, train_loader, test_loader)
