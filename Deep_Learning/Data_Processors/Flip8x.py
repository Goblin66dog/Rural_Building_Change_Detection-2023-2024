import cv2
import numpy as np
from osgeo import gdal
from Deep_Learning.Data_Readers.Data_Reader import Dataset
import os
import glob


class Flip8x:
    def __init__(self, image,image_shape="CHW"):
        super(Flip8x, self).__init__()
        self.image = image
        self.image_shape = image_shape

    def flip8x(self):
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
        self.image = np.transpose(self.image, axs[0])
        image_0     = self.image
        image_90    = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        image_180   = cv2.rotate(self.image, cv2.ROTATE_180)
        image_270   = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_0_r   = cv2.flip(image_0, 1)
        image_90_r  = cv2.flip(image_90, 0)
        image_180_r = cv2.flip(image_180, 1)
        image_270_r = cv2.flip(image_270, 0)

        image_0     =np.transpose(image_0    , axs[1])
        image_90    =np.transpose(image_90   , axs[1])
        image_180   =np.transpose(image_180  , axs[1])
        image_270   =np.transpose(image_270  , axs[1])
        image_0_r   =np.transpose(image_0_r  , axs[1])
        image_90_r  =np.transpose(image_90_r , axs[1])
        image_180_r =np.transpose(image_180_r, axs[1])
        image_270_r =np.transpose(image_270_r, axs[1])


        return image_0, image_90,image_180,image_270,image_0_r,image_90_r,image_180_r,image_270_r

    def extract_name(self,image_path):
        name = os.path.splitext(os.path.basename(image_path))[0]
        return name

    def flip(self, input_path, output_path):

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
        self.image = np.transpose(self.image, axs[0])

        name = ["_0","_90","_180","_270","_0_r","_90_r","_180_r","_270_r" ]
        name_num = 0
        for each_image in self.flip8x():
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(output_path+"\\"+self.extract_name() + str(name[name_num])+".TIF",
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
        return self.image


if __name__ == "__main__":
    pack_path = r"D:\Project\CUMT_PAPER_DATASETS"
    pack_path_list = glob.glob(os.path.join(pack_path, "*"))

    image_c = ["CHW", "HW", "HW"]
    i = 0
    for each_pack in pack_path_list:
        data_type = os.path.splitext(os.path.basename(each_pack))[0]
        image_path = glob.glob(os.path.join(pack_path, data_type+"/*"))

        for each_path in image_path:
            Flip8x(each_path,
                   r"D:\Project\CUMT_PAPER_DATASETS_FINAL"+"\\"+data_type,
                   image_c[i],
                   ).flip()
        i+=1