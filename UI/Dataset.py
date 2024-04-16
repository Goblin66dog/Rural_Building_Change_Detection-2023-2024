from osgeo import gdal
# Dataset类:
class Dataset:
    # 初始化（图像栅格、图像shape、投影、栅格数据类型）
    def __init__(self, file_path):
        # 读取影像
        self.file_path = file_path
        self.data = gdal.Open(self.file_path)
        # 读取图像列数(宽度)
        self.cols = self.data.RasterXSize
        # 读取图像行数(高度)
        self.rows = self.data.RasterYSize
        # 地图投影信息
        self.proj = self.data.GetProjection()
        # 仿射变换参数
        self.trans = self.data.GetGeoTransform()
        self.min_x = self.trans[0]
        self.max_y = self.trans[3]
        self.pixel_width = self.trans[1]
        self.pixel_height = self.trans[5]
        self.max_x = self.min_x + (self.cols * self.pixel_width)
        self.min_y = self.max_y + (self.rows * self.pixel_height)
        # 读取图像栅格数据
        self.array = self.data.ReadAsArray(0, 0, self.cols, self.rows)

        # 图像维度/波段
        if len(self.array.shape) == 2:
            self.bands = 1
        else:
            self.bands = self.array.shape[0]
        # 判断栅格数据的类型
        if 'int8' in self.array.dtype.name:
            self.type = gdal.GDT_Byte
        elif 'int16' in self.array.dtype.name:
            self.type = gdal.GDT_UInt16
        else:
            self.type = gdal.GDT_Float32

        # 释放内存，如果不释放，在arcgis，envi中打开该图像时会显示文件被占用
        del self.data