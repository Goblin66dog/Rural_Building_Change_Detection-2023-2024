import numpy as np
import cv2

class Padding:
    def __init__(self, image, image_shape="CHW"):
        super(Padding,self).__init__()
        self.image = image
        self.image_shape = image_shape
    def mirror_padding(self, target_height, target_width):



        while target_height > self.image.shape[0] or target_width > self.image.shape[1]:
            len_h_add = target_height - self.image.shape[0]
            len_w_add = target_width - self.image.shape[1]

            if len_w_add * len_h_add:
                image_mirror_add_w = self.image[:, self.image.shape[1] - len_w_add:]
                image_mirror_add_w = cv2.flip(image_mirror_add_w, 1)

                image_mirror_add_h = self.image[self.image.shape[0] - len_h_add:, :]
                image_mirror_add_h = cv2.flip(image_mirror_add_h, 0)

                image_mirror_add = self.image[self.image.shape[0] - len_h_add:, self.image.shape[1] - len_w_add:]
                image_mirror_add = cv2.flip(image_mirror_add, -1)

                image = np.concatenate([self.image, image_mirror_add_w], axis=1)
                image_mirror_add_h = np.concatenate([image_mirror_add_h, image_mirror_add], axis=1)
                self.image = np.concatenate([image, image_mirror_add_h], axis=0)

            elif len_w_add:

                image_mirror_add_w = self.image[:, self.image.shape[1] - len_w_add:]
                image_mirror_add_w = cv2.flip(image_mirror_add_w, 1)

                self.image = np.concatenate([self.image, image_mirror_add_w], axis=1)

            elif len_h_add:

                image_mirror_add_h = self.image[self.image.shape[0] - len_h_add:, :]
                image_mirror_add_h = cv2.flip(image_mirror_add_h, 0)

                self.image = np.concatenate([self.image, image_mirror_add_h], axis=0)

            else:
                pass
    def normal_padding_2Dim(self, target_height, target_width):
        # 计算图像Padding高度
        len_h_add = target_height - self.image.shape[0]
        # 计算图像Padding宽度
        len_w_add = target_width - self.image.shape[1]
        # padding
        self.image = np.pad(self.image, ((0, len_h_add), (0, len_w_add)), "constant", constant_values=(0, 0))
    def normal_padding_3Dim(self, target_height, target_width):
        # 计算图像Padding高度
        len_h_add = target_height - self.image.shape[0]
        # 计算图像Padding宽度
        len_w_add = target_width - self.image.shape[1]
        # padding
        self.image = np.pad(self.image, ((0, len_h_add), (0, len_w_add),(0,0)), "constant", constant_values=(0, 0))
    def normal_padding_4Dim(self, target_height, target_width):
        # 计算图像Padding高度
        len_h_add = target_height - self.image.shape[0]
        # 计算图像Padding宽度
        len_w_add = target_width - self.image.shape[1]
        # padding
        self.image = np.pad(self.image, ((0, len_h_add), (0, len_w_add),(0,0),(0,0)), "constant", constant_values=(0, 0))
    def mini_padding_2Dim(self, divide):
        image_h = self.image.shape[0]
        image_w = self.image.shape[1]
        len_h_add = 0
        len_w_add = 0

        while image_h % divide != 0:
            len_h_add += 1
            image_h += 1

        while image_w % divide != 0:
            len_w_add += 1
            image_w += 1

        self.image = np.pad(self.image, ((0, len_h_add), (0, len_w_add)), "constant", constant_values=(0, 0))
    def mini_padding_3Dim(self, divide):
        image_h = self.image.shape[0]
        image_w = self.image.shape[1]
        len_h_add = 0
        len_w_add = 0

        while image_h % divide != 0:
            len_h_add += 1
            image_h += 1

        while image_w % divide != 0:
            len_w_add += 1
            image_w += 1

        self.image = np.pad(self.image, ((0, len_h_add), (0, len_w_add),(0,0)), "constant", constant_values=(0, 0))
    def mini_padding_4Dim(self, divide):
        image_h = self.image.shape[0]
        image_w = self.image.shape[1]
        len_h_add = 0
        len_w_add = 0

        while image_h % divide != 0:
            len_h_add += 1
            image_h += 1

        while image_w % divide != 0:
            len_w_add += 1
            image_w += 1

        self.image = np.pad(self.image, ((0, len_h_add), (0, len_w_add),(0,0),(0,0)), "constant", constant_values=(0, 0))

    def mir(self, target_height, target_width):
        if self.image_shape == "CHW":
            self.image = np.transpose(self.image, [1, 2, 0])#HWC
            self.mirror_padding(target_height, target_width)
            self.image = np.transpose(self.image, [2, 0, 1])#CHW
            return self.image
        elif self.image_shape == "WH":
            self.image = np.transpose(self.image, [1, 0])#HW
            self.mirror_padding(target_height, target_width)
            self.image = np.transpose(self.image, [1, 0])#WH
            return self.image
        elif self.image_shape == "CWH":
            self.image = np.transpose(self.image, [2, 1, 0])#HWC
            self.mirror_padding(target_height, target_width)
            self.image = np.transpose(self.image, [2, 1, 0])#CWH
            return self.image
        elif self.image_shape == "WHC":
            self.image = np.transpose(self.image, [1, 0, 2])#HWC
            self.mirror_padding(target_height, target_width)
            self.image = np.transpose(self.image, [1, 0, 2])#WHC
            return self.image
        elif self.image_shape == "NCHW":
            self.image = np.transpose(self.image, [2, 3, 0, 1])#HWNC
            self.mirror_padding(target_height, target_width)
            self.image = np.transpose(self.image, [2, 3, 0, 1])#NCHW
            return self.image
        else:
            self.mirror_padding(target_height, target_width)
            return self.image
    def nor(self, target_height, target_width):
        if self.image_shape == "CHW":
            self.image = np.transpose(self.image, [1, 2, 0])#HWC
            self.normal_padding_3Dim(target_height, target_width)
            self.image = np.transpose(self.image, [2, 0, 1])#CHW
            return self.image
        elif self.image_shape == "WH":
            self.image = np.transpose(self.image, [1, 0])#HW
            self.normal_padding_2Dim(target_height, target_width)
            self.image = np.transpose(self.image, [1, 0])#WH
            return self.image
        elif self.image_shape == "CWH":
            self.image = np.transpose(self.image, [2, 1, 0])#HWC
            self.normal_padding_3Dim(target_height, target_width)
            self.image = np.transpose(self.image, [2, 1, 0])#CWH
            return self.image
        elif self.image_shape == "WHC":
            self.image = np.transpose(self.image, [1, 0, 2])#HWC
            self.normal_padding_3Dim(target_height, target_width)
            self.image = np.transpose(self.image, [1, 0, 2])#WHC
            return self.image
        elif self.image_shape == "NCHW":
            self.image = np.transpose(self.image, [2, 3, 0, 1])#HWNC
            self.normal_padding_4Dim(target_height, target_width)
            self.image = np.transpose(self.image, [2, 3, 0, 1])#NCHW
            return self.image
        else:
            self.normal_padding_2Dim(target_height, target_width)
            return self.image

    def min(self, divide):
        if self.image_shape == "CHW":
            self.image = np.transpose(self.image, [1, 2, 0])#HWC
            self.mini_padding_3Dim(divide)
            self.image = np.transpose(self.image, [2, 0, 1])#CHW
            return self.image
        elif self.image_shape == "WH":
            self.image = np.transpose(self.image, [1, 0])#HW
            self.mini_padding_2Dim(divide)
            self.image = np.transpose(self.image, [1, 0])#WH
            return self.image
        elif self.image_shape == "CWH":
            self.image = np.transpose(self.image, [2, 1, 0])#HWC
            self.mini_padding_3Dim(divide)
            self.image = np.transpose(self.image, [2, 1, 0])#CWH
            return self.image
        elif self.image_shape == "WHC":
            self.image = np.transpose(self.image, [1, 0, 2])#HWC
            self.mini_padding_3Dim(divide)
            self.image = np.transpose(self.image, [1, 0, 2])#WHC
            return self.image
        elif self.image_shape == "NCHW":
            self.image = np.transpose(self.image, [2, 3, 0, 1])#HWNC
            self.mini_padding_4Dim(divide)
            self.image = np.transpose(self.image, [2, 3, 0, 1])#NCHW
            return self.image
        else:
            self.mini_padding_2Dim(divide)
            return self.image

if __name__ == "__main__":
    from Deep_Learning.Data_Readers.Data_Reader import Dataset
    image = Dataset(r"D:\Wuhan_University_Datasets\image\0349.jpg").array
    image = Padding(image).min(17)
    image = np.transpose(image,[1,2,0])
    print(image.shape)
    cv2.imshow("image",image)
    cv2.waitKey(0)



