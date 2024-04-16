import numpy as np
import cv2


#分通道进行flip & padding & normalize
class PercentLinearEnhancement:
    def __init__(self, image, min_out=None, max_out=None , clip_num = 2,image_shape="CHW"):
        super(PercentLinearEnhancement, self).__init__()
        self.image = image
        self.image_shape = image_shape
        self.clip_num = clip_num
        self.min_out, self.max_out = min_out, max_out

    def gray_process(self):
        high_value = np.percentile(self.image, 100 - self.clip_num)  # 取得98%直方图处对应灰度
        low_value = np.percentile(self.image, self.clip_num)  # 取得2%直方图处对应灰度
        if self.min_out and self.max_out:
            high_value = self.max_out
            low_value  = self.min_out
        truncated_gray = np.clip(self.image, a_min=low_value, a_max=high_value)
        processed_gray = ((truncated_gray - low_value) / (high_value - low_value)) * (high_value - low_value)
        return processed_gray


if __name__ == "__main__":
    from Deep_Learning.Data_Readers.Data_Reader import Dataset

    image = Dataset(r"D:\Project\CUMT_PAPER_DATASETS\texture\72_22_4.TIF").array
    label = Dataset(r"D:\Project\CUMT_PAPER_DATASETS\texture\72_20_0.TIF").array
    image = PercentLinearEnhancement(image,clip_num=2).gray_process()
    image = cv2.normalize(image, label, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # label = np.transpose(label, [1, 2, 0])
    print(image.shape)
    cv2.imshow("image",label)
    cv2.waitKey(0)

