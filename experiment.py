import os
from torch.utils.data import DataLoader

from train import Trainer
from dataset import BeautyGlowDataSet
from models import BeautyGlow


beauty_glow_dataset = BeautyGlowDataSet(
    source_img_dir=os.path.abspath("./data/source"),
    reference_img_dir=os.path.abspath("./data/reference")
)

data_loader = DataLoader(beauty_glow_dataset, batch_size=100, shuffle=True)
trainer = Trainer(
    data_loader
)
trainer.train(100)
