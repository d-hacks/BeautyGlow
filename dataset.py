from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision import transforms

import glow

class BeautyGlowDataSet(Dataset):

    def __init__(self, source_img_dir, reference_img_dir):
        self.glow = glow.Glow(3, 32, 4, affine=True, conv_lu=True)
        self.source_file_list = glob.glob(source_img_dir + "/*.jpg")
        self.reference_file_list = glob.glob(reference_img_dir + "/*.jpg")
        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        l_x_list = []
        for source_file in self.source_file_list:
            source_img = Image.open(source_file)
            source_img_tensor = self.transforms(source_img)[None, :, :, :]
            _l_x = self.glow(source_img_tensor)
            l_x_list.append(_l_x)

        self.l_x = torch.stack(l_x_list).mean()

        l_y_list = []
        for reference_file in self.reference_file_list:
            reference_img = Image.open(reference_file)
            reference_img_tensor = self.transforms(reference_img)[None, :, :, :]
            _l_y = self.glow(reference_img_tensor)
            l_y_list.append(_l_y)

        self.l_y = torch.stack(l_y_list).mean()

        def __getitem__(self, index):
            source_img_path = self.source_file_list[index]
            source_img = Image.open(source_img_path)
            source_img_tonsor = self.transforms(source_img)

            reference_img_path = self.reference_file_list[index]
            reference_img = Image.open(reference_img_path)
            reference_img_tonsor = self.transforms(reference_img)

            return source_img_tonsor, reference_img_tensor

        def __len__(self):
            return len(self.source_file_list)
