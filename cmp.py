from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification
from ptflops import get_model_complexity_info
import torch 
from PIL import Image
import requests

import torchvision
from net.cvt_offical import BiseCvt_same_params,Base_cvt_cifar,BiseCvt_same_flops
from net.wrn import ResNet50_for_cifar

pth = 'CvT-13-224x224-IN-1k.pth'

model = Base_cvt_cifar(pth)
flops, params = get_model_complexity_info(model,(3,32,32),print_per_layer_stat=False)
print('base','flops',flops, 'params',params)

model = BiseCvt_same_params(pth)
flops, params = get_model_complexity_info(model,(3,32,32),print_per_layer_stat=False)
print('bise_same_params','flops',flops, 'params',params)

model = BiseCvt_same_flops(pth)
flops, params = get_model_complexity_info(model,(3,32,32),print_per_layer_stat=False)
print('bise_same_flops','flops',flops, 'params',params)

model = ResNet50_for_cifar()
flops, params = get_model_complexity_info(model,(3,224,224),print_per_layer_stat=False)
print('resnet50','flops',flops, 'params',params)