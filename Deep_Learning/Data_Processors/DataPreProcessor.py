import cv2
import numpy as np
from osgeo import gdal
from Deep_Learning.Data_Readers.Data_Reader import Dataset
import os
import glob
from Deep_Learning.Data_Processors.Padding import Padding
from Deep_Learning.Data_Processors.Percent_Linear_Enhancement import PercentLinearEnhancement
from Deep_Learning.Data_Processors.Flip8x import Flip8x

# 1.linear enhance
# 2.padding
# 3.flip
#

class Processor:
    def __init__(self, input_path,output_path, image_shape="CHW", mode="LPF",target_h=256,target_w=256):
        super(Processor, self).__init__()
        self.item = Dataset(input_path)
        self.output_path = output_path
        self.image = self.item.array
        self.image_shape = image_shape
        self.mode = mode


    def extract_name(self):
        name = os.path.splitext(os.path.basename(self.item.file_path))[0]
        return name


    def processor(self):
        if self.image_shape == "CHW":
            axs = [1,2,0],[2,0,1]
        elif self.image_shape == "WH":
            axs = [1, 0],[1, 0]
        elif self.image_shape == "CWH":
            axs = [2, 1, 0],[2, 1, 0]
        elif self.image_shape == "WHC":
            axs = [1, 0, 2],[1, 0, 2]
        elif self.image_shape == "NCHW":
            axs = [2, 3, 0, 1],[2, 3, 0, 1]
        else:
            axs = [0,1],[0,1]

        if self.mode == "LPF":
            self.image = PercentLinearEnhancement(self.image, image_shape=self.image_shape).gray_process()
            self.image = Padding(self.image, image_shape=self.image_shape).nor(256,256)
            self.image = Flip8x(self.image, image_shape=self.image_shape).flip8x()
        elif self.mode == "PF":
            self.image = Padding(self.image, image_shape=self.image_shape).nor(256,256)
            self.image = Flip8x(self.image, image_shape=self.image_shape).flip8x()
        elif self.mode == "LP":
            self.image = PercentLinearEnhancement(self.image, image_shape=self.image_shape).gray_process()
            self.image = Padding(self.image, image_shape=self.image_shape).nor(256,256)
        elif self.mode == "P":
            self.image = Padding(self.image, image_shape=self.image_shape).nor(256,256)
        elif self.mode == "L":
            self.image = PercentLinearEnhancement(self.image, image_shape=self.image_shape).gray_process()

        name = ["_0", "_90", "_180", "_270", "_0_r", "_90_r", "_180_r", "_270_r"]
        name_num = 0
        if "F" in self.mode:
            for each_image in self.image:
                each_image = np.transpose(each_image, axs[0])
                driver = gdal.GetDriverByName('GTiff')
                dataset = driver.Create(self.output_path + "\\" + self.extract_name() + str(name[name_num]) + ".TIF",
                                        each_image.shape[1],
                                        each_image.shape[0],
                                        self.item.bands,
                                        self.item.type)
                each_image = np.transpose(each_image, axs[1])

                for each_bands in range(self.item.bands):
                    if self.item.bands == 1:
                        dataset.GetRasterBand(1).WriteArray(each_image)
                        continue
                    dataset.GetRasterBand(each_bands + 1).WriteArray(each_image[each_bands])
                dataset.FlushCache()  # 确保所有写入操作都已完成
                dataset = None
                name_num += 1
        else:
            self.image = np.transpose(self.image, axs[0])
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(self.output_path + "\\" + self.extract_name() + str(name[name_num]) + ".TIF",
                                    self.image.shape[1],
                                    self.image.shape[0],
                                    self.item.bands,
                                    self.item.type)
            self.image = np.transpose(self.image, axs[1])

            for each_bands in range(self.item.bands):
                if self.item.bands == 1:
                    dataset.GetRasterBand(1).WriteArray(self.image)
                    continue
                dataset.GetRasterBand(each_bands + 1).WriteArray(self.image[each_bands])
            dataset.FlushCache()  # 确保所有写入操作都已完成
            dataset = None
            name_num += 1

    @staticmethod
    def batch_processor(input_pack_path, output_pack_path,image_shape,mode):

        image_path = glob.glob(os.path.join(input_pack_path+"\*"))
        for each_path in image_path:
            Processor(each_path, output_pack_path,image_shape,mode=mode).processor()


if "__main__" == __name__:
    # if self.mode == "LPF":
    #     self.image = PercentLinearEnhancement(self.image, image_shape=self.image_shape).gray_process()
    #     self.image = Padding(self.image, image_shape=self.image_shape).nor(256, 256)
    #     self.image = Flip8x(self.image, image_shape=self.image_shape).flip8x()
    # elif self.mode == "PF":
    #     self.image = Padding(self.image, image_shape=self.image_shape).nor(256, 256)
    #     self.image = Flip8x(self.image, image_shape=self.image_shape).flip8x()
    # elif self.mode == "P":
    #     self.image = Padding(self.image, image_shape=self.image_shape).nor(256, 256)

    Processor.batch_processor(r"D:\Github_Repo\validate_ori_image\4.12(1)\label",
                              r"D:\Github_Repo\Deploy\label",
                              "HW",
                              "PF"
                              )
    # Processor.batch_processor(r"D:\Github_Repo\validate_ori_image\4.12(1)\image2",
    #                           r"D:\Github_Repo\Deploy\image2",
    #                           "CHW",
    #                           "LPF"
    #                           )



