import pickle
import torch.nn as nn
import torch
import cv2

input=torch.randn(1,1,400,200)
net=nn.Sequential(nn.Conv2d(1,3,7,2,3,bias=False))
output=net(input)

# file=pickle.load(open('results.pkl','rb'))
CLASSES = ('water',)
a={cat: i + 1 for i, cat in enumerate(CLASSES)}
print(a)
debug=1
# print(file)