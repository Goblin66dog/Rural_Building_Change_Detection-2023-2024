import cv2
import random
import numpy as np

class RandomFlip:
    def __init__(self, image, label, image_shape="CHW"):
        super(RandomFlip, self).__init__()
        self.image = image
        self.label = label
        self.image_shape = image_shape
        self.flip_code = random.choice([-1,0,1,2])

    def random_flip(self):
        if self.flip_code != 2:
            self.image = cv2.flip(self.image, self.flip_code)
            self.label = cv2.flip(self.label, self.flip_code)

    def flip(self):
        if self.image_shape == "CHW":
            self.image = np.transpose(self.image, [1, 2, 0])#HWC
            self.label = np.transpose(self.label, [1, 2, 0])#HWC
            self.random_flip()
            self.image = np.transpose(self.image, [2, 0, 1])#CHW
            self.label = np.transpose(self.label, [2, 0, 1])#CHW
        return self.image, self.label

if __name__ == "__main__":

    from Deep_Learning.Data_Readers.Data_Reader import Dataset
    image = Dataset(r"D:\Project\DeepLearning_logs\2024.3.24 Segformer(0.8Focal a=0.95, g=2 +0.2DICE + 200epoch + 2 batchsize + RMS)\CUMT_PAPER_DATASETS\image\19_21_0.TIF").array
    # # image[image > 0] = 255
    image, label = RandomFlip(image,image).flip()
    image = np.transpose(image, [1, 2, 0])
    cv2.imshow("image",image[:,:,:3])
    # cv2.imshow("label",label)
    cv2.waitKey(0)