#!/usr/bin/python
# coding: utf-8

import torch as t
import torchvision as tv
import matplotlib.pyplot as plt

batch_size=64
dataste=tv.datasets.MNIST(root='./mnist/',train=True,transform=tv.transforms.ToTensor())
dataloader=t.utils.data.DataLoader(dataset=dataste,batch_size=batch_size,shuffle=True,drop_last=True)
lr=0.0001
encoder_c=t.nn.Sequential(
t.nn.Conv2d(1,32,5,1,2),
t.nn.Softplus(),
t.nn.Conv2d(32,64,5,2,2),
t.nn.Softplus(),
t.nn.Conv2d(64,128,5,2,2),
t.nn.Softplus()
)
encoder_l=t.nn.Sequential(
t.nn.Linear(128*7*7,100),
t.nn.Softplus(),
t.nn.Linear(100,10),
t.nn.Sigmoid()
)
class EN(t.nn.Module):
    def __init__(self):
        super(EN,self).__init__()
        self.c=encoder_c
        self.l=encoder_l
    def forward(self,x):
        x=self.c(x)
        x=x.view(-1,128*7*7)
        x=self.l(x)
        return x
    
encoder=EN()

decoder_l=t.nn.Sequential(
t.nn.Linear(10,100),
t.nn.Softplus(),
t.nn.Linear(100,128*7*7),
t.nn.Softplus()
)
decoder_c=t.nn.Sequential(
t.nn.ConvTranspose2d(128,64,5,2,2),
t.nn.Softplus(),
t.nn.ConvTranspose2d(64,32,5,2,2,1),
t.nn.Softplus(),
t.nn.ConvTranspose2d(32,1,5,1,1),
t.nn.Sigmoid()
)
class DE(t.nn.Module):
    def __init__(self):
        super(DE,self).__init__()
        self.l=decoder_l
        self.c=decoder_c
    def forward(self,x):
        x=self.l(x)
        x=x.view(-1,128,7,7)
        x=self.c(x)
        return x

decoder=DE()
    
opt_en=t.optim.Adam(encoder.parameters(),lr=lr)
opt_de=t.optim.Adam(decoder.parameters(),lr=lr)

loss_func_en=t.nn.CrossEntropyLoss()
loss_func_de=t.nn.MSELoss()

def train(num=10):
    for i in xrange(int(num)):
        print("ecoph:%d" %i)
        for b_x,b_y in dataloader:
            b_x=t.autograd.Variable(b_x)
            b_y=t.autograd.Variable(b_y)
            p_en=encoder(b_x)
            p_de=decoder(p_en)
            loss_de=loss_func_de(p_de,b_x)
            loss_en=loss_func_en(p_en,b_y)
            opt_en.zero_grad()
            loss_en.backward(retain_variables=True)
            opt_en.step()
            opt_de.zero_grad()
            loss_de.backward()
            opt_de.step()
        
