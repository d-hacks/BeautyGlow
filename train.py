import torch.optim as optim
import torch
import torch.nn
import torch.nn.functional as F
import torchvision

from models import BeautyGlow


class Trainer:
    def __init__(self, data_loader):
        self.data_loader
        self.model = BeautyGlow()
        self.optim = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epoch):
        for e in range(epoch):
            for reference, source, l_x, l_y in self.data_loader:
                result, loss = self.model(reference, source, l_x, l_y)
                loss.backword()
                self.optim.step()

                print("epoch: {} loss:{}".format(e, loss))
                torchvision.utils.save_image(result, "results/{}_result.jpg".format(e))
