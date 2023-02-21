import torch

from src.img_gen.load_img import load_imgs, show_imgs
import random

import os

from src.img_gen.net import Net

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



learning_rate = 0.0005
epoches = 50
steps = 250


model = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
imgs, ogs = load_imgs(steps)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def train():
    for epoch in range(epoches):
        avg_cost = 0
        splited_imgs = imgs[0:(int(len(imgs)*(epoch+1)/epoches))]
        random.shuffle(splited_imgs)
        print('images: ' + str(len(splited_imgs)))
        for img in splited_imgs:
            optimizer.zero_grad()
            hypothesis = model(img[0])
            cost = criterion(hypothesis, img[1])  # <= compute the loss function
            avg_cost += cost.item()
            cost.backward()  # <= compute the gradient of the loss/cost function
            optimizer.step()
        avg_cost /= epoches
        if epoch % int(epoches/5) == int(epoches/5)-1:
            print('avg_cost: ' + str(avg_cost))

    print('__________________')

show_imgs(imgs[-1][0])
for j in range(10):
    train()
    for o in range(len(ogs)):
        for i in range(steps):
            ogs[o] = model.forward(ogs[o])
    show_imgs(ogs)



