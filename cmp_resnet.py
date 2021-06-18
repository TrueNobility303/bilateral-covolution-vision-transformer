from os import X_OK
import torch 
import torchvision 

from net.cvt_offical import get_cvt13_pretrained
from utils import train_transform, test_transform,initialize,smooth_crossentropy
from config import device
from ptflops import get_model_complexity_info
import torch.nn.functional as F 
import torch.nn as nn

initialize(42)

device = torch.device('cuda:0')
model = torchvision.models.resnet50(pretrained=True) 
flops, params = get_model_complexity_info(model,(3,224,224),print_per_layer_stat=False)
print('flops',flops, 'params',params)

model.fc = nn.Linear(512*4,10)
model.to(device)
# flops 4.06 GMac params 19.6 M 均比resnet50小

BATCH = 32
trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

LR = 1e-3
DECAY = 0
EPOCH = 20

optimizer = torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=DECAY)
def train(dataloader,train_sam=False):
    model.train()
    tot_loss = 0
    tot_num = 0
    for i,data in enumerate(dataloader):
        x,y = data
        x = x.to(device)
        y = y.to(device)

        x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=True)        
        logits = model(x)

        #非SAM
        if train_sam is False:
            optimizer.zero_grad()
            loss = smooth_crossentropy(logits,y).mean()
            loss.backward()
            optimizer.step()

        #可以考虑使用SAM进行训练
        else:
            loss = smooth_crossentropy(logits,y).mean()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            smooth_crossentropy(model(x), y).mean().backward()
            optimizer.second_step(zero_grad=True)

        tot_loss += loss.item()
        tot_num += x.shape[0]

    return tot_loss / tot_num

def test(dataloader):
    model.eval()
    correct = 0
    num = 0
    for i,data in enumerate(dataloader):
        x,y = data
        x = x.to(device)
        y = y.to(device)
        x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=True)  

        logits = model(x)
        pred = logits.argmax(1)
        correct += torch.sum(pred==y).item()
        num += x.shape[0]
    acc = correct / num
    return acc 

if __name__ == '__main__':
    accs = []
    losses = []
    model_path = 'pth/resnet50.pth'
    for epoch in range(EPOCH):
        #训练sam大叔优化器,也可以考虑使用KFAC优化器
        loss = train(trainloader)
        testacc = test(testloader)
        accs.append(testacc)
        losses.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
        
        #可以考虑使用scheduler
        #scheduler(epoch)
    
        torch.save(model.state_dict(),model_path)
    