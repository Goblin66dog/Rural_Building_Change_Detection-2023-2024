import cv2
import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np
from osgeo import gdal

from Deep_Learning.Models.UNet.model import                     UNet
from Deep_Learning.Models.SegFormer.model import                SegFormer
from Deep_Learning.Models.SegFormer_OutConv.model import        SegFormerOutConv



import warnings
warnings.filterwarnings("ignore")


class DeployByPth:
    def __init__(self, model, image_path, model_path,output_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(device=self.device)
        self.model = model.eval()
        self.image_path = image_path
        self.output_path = output_path

    @staticmethod
    def extract_name(file_path):
        name = os.path.splitext(os.path.basename(file_path))[0]
        return name

    def SaveWithGeoInfo(self, image_array,image_path,image_shape="HW"):
        if image_shape == "CHW":
            axs = [1,2,0],[2,0,1]
        elif image_shape == "WH":
            axs = [1, 0],[1, 0]
        elif image_shape == "CWH":
            axs = [2, 1, 0],[2, 1, 0]
        elif image_shape == "WHC":
            axs = [1, 0, 2],[1, 0, 2]
        elif image_shape == "NCHW":
            axs = [2, 3, 0, 1],[2, 3, 0, 1]
        else:
            axs = [0,1],[0,1]

        image = np.transpose(image_array, axs[0])
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(self.output_path + "\\" + self.extract_name(file_path=image_path)+ "prediction.TIF",
                                image.shape[1],
                                image.shape[0],
                                1,
                                gdal.GDT_Float32)
        image = np.transpose(image, axs[1])
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        dataset.GetRasterBand(1).WriteArray(image)
        dataset.FlushCache()  # 确保所有写入操作都已完成
        dataset = None

    def I1(self):
        step = 0
        dataloader = Deploy_Reader_I1.DataLoader(self.image_path)
        data = torch.utils.data.DataLoader(
            dataset=dataloader,
            batch_size=1,
            shuffle=False
        )
        for image, path in data:
            image = image.to(device=self.device, dtype=torch.float32)

            prediction = self.model(image)

            prediction = torch.sigmoid(prediction)

            prediction[prediction > 0.5] = 1
            prediction[prediction <= 0.5] = 0

            prediction = np.array(prediction.data.cpu())
            self.SaveWithGeoInfo(prediction, path)

            step += 1

    def I2(self):
        step = 0
        dataloader = Deploy_Reader_I2.DataLoader(self.image_path)
        data = torch.utils.data.DataLoader(
            dataset=dataloader,
            batch_size=1,
            shuffle=False
        )
        for image1,image2, path in data:
            image1 = image1.to(device=self.device, dtype=torch.float32)
            image2 = image2.to(device=self.device, dtype=torch.float32)

            prediction = self.model(torch.cat([image1, image2], dim=1).to(device=self.device, dtype=torch.float32),
                                    image2)
            # prediction = self.model(image1, image2)

            prediction = torch.sigmoid(prediction)

            prediction[prediction > 0.05] = 1
            prediction[prediction <= 0.05] = 0

            prediction = np.array(prediction.data.cpu())
            self.SaveWithGeoInfo(prediction, path)
            step += 1

if __name__ == "__main__":
    model1 = UNet(in_channels=5,num_classes=1)
    model2 = AGUNet(in_channels=5,num_classes=1)
    model3 = ASPPU2Net(image_channels=4,texture_channels=1,num_classes=1)
    model4 = DeepLab(in_channels=5, num_classes=1)
    model5 = SegFormer(in_channels=5,num_classes=1,backbone="b3")
    model6 = SegFormerOutConv(in_channels=5,num_classes=1,backbone="b3")
    model7 = SegFormerUNet(in_channels=5,num_classes=1,backbone="b3")
    model8 = SegFormerUNetConcise(in_channels=5,num_classes=1,backbone="b3")
    image_path = r"D:\Github_Repo\validate_ori_image\4.13"
    model_path = (r"D:\Github_Repo\logs\SegFormer_U\model logs\model2.pth")
    DeployByPth(model8, image_path, model_path,).I2()
