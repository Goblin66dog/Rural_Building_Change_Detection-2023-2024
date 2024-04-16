import glob
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from Deep_Learning.Data_Processors.Padding import Padding

from Deep_Learning.Data_Readers import Data_Reader
import cv2

class DataLoader(Dataset):
    def __init__(self, input_datasets_path):
        super(DataLoader, self).__init__()
        self.image1_path = glob.glob(os.path.join(input_datasets_path, "image/*"))

    def __getitem__(self, index):

        #根据index读取图像和标签
        image1_path = self.image1_path[index]

        #读取训练图片和标签图片
        image1 = Data_Reader.Dataset(image1_path)

        image1_array = image1.array
        image1_width = image1.width
        image1_height= image1.height
        image1_bands = image1.bands
        del image1


        #reshape
        image1_array = image1_array.reshape(image1_bands, image1_height, image1_width)
        image1_array = image1_array[::-1, ]
        image1_array = Padding(image1_array, image_shape="CHW").nor(256, 256)


        image1_array = cv2.normalize(image1_array,None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # 转为tensor
        image1_array = torch.tensor(image1_array,dtype=torch.float32)
        return image1_array, self.image1_path

    def __len__(self):
        return len(self.image1_path)

if __name__ == "__main__":

    train_dataloader = DataLoader(r"D:\Project\CUMT_PAPER_DATASETS_FINAL")
    train_data = torch.utils.data.DataLoader(
        dataset=train_dataloader,
        batch_size=1,
        shuffle=True
    )
    # # writer = {
    # #     "data":SummaryWriter(r"logs")
    # # }
    step = 1
    for x, y, z in train_data:
        print(x.shape, step)
        # writer["data"].add_image1s("image1", image1, step)
        # writer["data"].add_image1s("label", label, step)

        step += 1
    # # writer["data"].close()

