import glob
import os

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

import Data_Reader


class DataLoader(Dataset):
    def __init__(self, input_datasets_path):
        super(DataLoader, self).__init__()
        self.image_path = glob.glob(os.path.join(input_datasets_path, "image/*"))
        self.label_path = glob.glob(os.path.join(input_datasets_path, "label/*"))

    def __getitem__(self, index):

        #根据index读取图像和标签
        image_path = self.image_path[index]
        label_path = self.label_path[index]

        #读取训练图片和标签图片
        image = Data_Reader.Dataset(image_path)
        label = Data_Reader.Dataset(label_path)

        image_array = image.array
        image_width = image.width
        image_height= image.height
        image_bands = image.bands
        del image
        label_array = label.array
        label_width = label.width
        label_height= label.height
        label_bands = label.bands
        del label

        #reshape
        image_array = image_array.reshape(image_bands, image_width, image_height)
        label_array = label_array.reshape(label_bands, label_width, label_height)

        # 转为tensor
        image_array = torch.tensor(image_array)
        label_array = torch.tensor(label_array)

        return image_array, label_array

    def __len__(self):
        return len(self.image_path)

if __name__ == "__main__":

    train_dataloader = DataLoader(r"D:\Wuhan_University_Datasets")
    train_data = torch.utils.data.DataLoader(
        dataset=train_dataloader,
        batch_size=1,
        shuffle=True
    )
    # # writer = {
    # #     "data":SummaryWriter(r"logs")
    # # }
    step = 1
    for x, y in train_data:
        print(x.shape, step)
        # writer["data"].add_images("image", image, step)
        # writer["data"].add_images("label", label, step)

        step += 1
    # # writer["data"].close()

