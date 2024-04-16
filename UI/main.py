import sys
from PyQt5.Qt import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import onnxruntime
from Dataset import Dataset
from osgeo import gdal
import os
import glob

class UiMain(QWidget):
    def __init__(self, onnx_model_file_path, master=None):
        super().__init__(master)
        self.image_files = None
        self.onnx_model_file_path = onnx_model_file_path
        self.sess          = None
        self.output_name   = None
        self.single_array  = None
        self.single_bands  = None
        self.single_width  = None
        self.single_height = None
        self.single_proj   = None
        self.single_trans  = None
        self.input_image_name   = None
        self.whether_finished   = None
        self.onnx_input_shape = 3, 512, 512
        ########全局窗口设置
        self.setWindowTitle("乡村建筑物提取程序")
        self.setFixedSize(960, 720)

        self.graphicsPreview = QGraphicsView()
        self.graphicsResview = QGraphicsView()

        self.init_onnx()
        self.init_ui()
    def init_ui(self):
        ########布局器
        stacked_layout        = QStackedLayout()
        container             = QVBoxLayout()
        tab_single_container  = QVBoxLayout()
        tab_batch_container   = QVBoxLayout()
        #Tab_single
        ##1
        tab_single_button_layout = QHBoxLayout()
        tab_single_bar_layout    = QHBoxLayout()
        ##2
        tab_single_tools_layout  = QHBoxLayout()
        tab_single_graph_layout  = QHBoxLayout()

        #Tab_batch
        ##1
        tab_batch_button_layout = QHBoxLayout()
        tab_batch_bar_layout    = QHBoxLayout()

        
        ########设置控件
        tab_bar           = QTabBar(parent=None)
        tab_single_widget = QWidget(self)
        tab_batch_widget  = QWidget(self)

        tab_bar.addTab("常规图像处理")
        tab_bar.addTab("批量文件处理")

        #Tab常规图像处理
        ##1
        tab_single_button_box      = QGroupBox("处理工具")
        tab_single_bar_box         = QGroupBox("处理进度")
        ##2
        tab_single_tools_box       = QGroupBox("处理模块")
        tab_single_graph_box       = QGroupBox("图像显示")

        tab_single_openfile_button = QPushButton("打开文件")
        tab_single_process_button  = QPushButton("运行")
        tab_single_savefile_button = QPushButton("保存文件")
        tab_single_progress_bar    = QProgressBar(tab_single_bar_box)
        #Tab批量文件处理
        tab_batch_button_box       = QGroupBox("处理图像")
        tab_batch_openfile_button  = QPushButton("打开文件夹")
        tab_batch_process_button   = QPushButton("批量处理")
        ########设置控件属性
        tab_single_progress_bar.setStyleSheet("QProgressBar {"
                                              "border: 2px solid grey; "
                                              "border-radius: 5px; "
                                              "color: rgb(20,20,20);  "
                                              "background-color: #FFFFFF; "
                                              "text-align: center;}"
                                              "QProgressBar::chunk {"
                                              "background-color: rgb(100,200,200);"
                                              "border-radius: 10px; margin: 0.1px;"
                                              "width: 1px;}")

        ########控件添加到布局器
        #Tab_single
        ##1
        tab_single_button_layout.addWidget(tab_single_openfile_button)
        tab_single_button_layout.addWidget(tab_single_process_button)
        tab_single_button_layout.addWidget(tab_single_savefile_button)
        tab_single_button_box   .setLayout(tab_single_button_layout)

        tab_single_bar_layout   .addWidget(tab_single_progress_bar)
        tab_single_bar_box      .setLayout(tab_single_bar_layout)
        ###2
        tab_single_tools_layout .addWidget(tab_single_button_box, Qt.AlignJustify)
        tab_single_tools_layout .addWidget(tab_single_bar_box,    Qt.AlignJustify)
        tab_single_tools_box    .setLayout(tab_single_tools_layout)

        tab_single_graph_layout.addWidget(self.graphicsPreview)
        tab_single_graph_layout.addWidget(self.graphicsResview)
        tab_single_graph_box   .setLayout(tab_single_graph_layout)
        ####3
        tab_single_container.addWidget(tab_single_graph_box)
        tab_single_container.addWidget(tab_single_tools_box)
        tab_single_widget   .setLayout(tab_single_container)

        #Tab_batch
        tab_batch_button_layout.addWidget(tab_batch_openfile_button)
        tab_batch_button_layout.addWidget(tab_batch_process_button)

        tab_batch_button_box.setLayout(tab_batch_button_layout)

        tab_batch_container.addWidget(tab_batch_button_box)

        tab_batch_widget  .setLayout(tab_batch_container)

        #Top_layer
        stacked_layout.addWidget(tab_single_widget)
        stacked_layout.addWidget(tab_batch_widget)
        container.addWidget(tab_bar)
        container.addLayout(stacked_layout)
        self.setLayout(container)

        #Tab_single


        ########绑定事件
        tab_bar.currentChanged[int]       .connect(stacked_layout.setCurrentIndex)
        tab_single_openfile_button.clicked.connect(self.open_single_image)
        tab_single_process_button.clicked .connect(self.process_single_image)
        tab_single_savefile_button.clicked.connect(self.save_single_image)

        tab_batch_openfile_button.clicked.connect(self.open_batch_image)
        tab_batch_process_button.clicked.connect(self.process_batch_image)
    # 事件
    ### 返回一个与graphicview大小相似的数组
    def return_resize(self, max_h, max_w):
        if self.single_height > self.single_width:
            changed_h = int(max_h)
            changed_w = int(self.single_width * (max_h / self.single_height))
        elif self.single_height < self.single_width:
            changed_h = int(self.single_height * (max_w / self.single_width))
            changed_w = int(max_w)
        else:
            changed_h = changed_w = max_h
        return changed_h, changed_w

    ### 根据波段数量返回相应的pixmap数组
    def warp_channels(self):
        if self.single_bands == 1:
            pixmap = self.single_array
        elif self.single_bands > 3:
            pixmap = np.transpose(self.single_array[:3], [1, 2, 0])
        else:
            pixmap = np.transpose(self.single_array, [1, 2, 0])
        return pixmap

    ### 初始化onnx模型
    def init_onnx(self):
        self.sess = onnxruntime.InferenceSession(self.onnx_model_file_path)
        self.input_image_name = self.sess.get_inputs()[0].name
        # self.input_texture_name = self.sess.get_inputs()[1].name
        self.output_name = self.sess.get_outputs()[0].name

    ### 运行打开文件函数，若filedialog返回的非0，则进行一系列预处理以显示预览图像
    def open_batch_image(self):
        image_files = []
        # 获取文件夹中的所有图像文件
        folder_path = QFileDialog.getExistingDirectory(self, "请选择包含图像的文件夹", '.', QFileDialog.ShowDirsOnly)
        image_files += glob.glob(os.path.join(folder_path, '*.tiff'))
        image_files += glob.glob(os.path.join(folder_path, '*.tif'))
        self.image_files = image_files
        print(self.image_files)
        return self.image_files
    def process_batch_image(self):

        # 选择输出的文件夹
        folder_output = QFileDialog.getExistingDirectory(self, "请选择输出的文件夹", '.', QFileDialog.ShowDirsOnly)

        for image in self.image_files:
            print(image)
            self.input_image(image)
            filename = os.path.basename(image)
            image_path = folder_output + "/" +filename
            print(image_path)
            self.process_single_image()

            self.save_image(image_path)


        QMessageBox.information(self, "File Saved!", "文件保存完成！")



    def input_image(self, file_name):
        try:
            # 预览
            if file_name != '':
                # 实例化image类
                image_object = Dataset(file_name)
                # 提取image信息
                self.single_array = image_object.array
                self.single_bands = image_object.bands
                self.single_width = image_object.cols
                self.single_height = image_object.rows
                self.single_proj = image_object.proj
                self.single_trans = image_object.trans
                del image_object

                # 设置resize大小（根据图像进行等比缩放）
                changed_h, changed_w = self.return_resize(430, 430)
                # 设置预览通道数  1、判定波段数量，对通道数为1的图像进灰度显示；对通道数大于4的图像进行伪彩色处理，最后一个通道为alpha通道
                #              2、将数值类型转化为uint8，标准化
                #              3、等比缩放，重采样）
                pixmap = self.warp_channels()
                pixmap = cv2.normalize(pixmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                pixmap = cv2.resize(pixmap, (changed_w, changed_h), interpolation=cv2.INTER_NEAREST)
                # cv2.cvtColor(pixmap, cv2.COLOR_BGR2RGB)
                # 预览（整个预览过程不会改变原数据）
                pixmap = QImage(pixmap, changed_w, changed_h, changed_w * min(3, self.single_bands),
                                (QImage.Format_Grayscale8, QImage.Format_RGB888)[min(2, self.single_bands) - 1])
                pixmap = QPixmap(pixmap)
                pixmap = QGraphicsPixmapItem(pixmap)
                scene = QGraphicsScene()
                scene.addItem(pixmap)
                self.graphicsPreview.setScene(scene)
                ########################################################################################################################
                cols = self.single_array.shape[1] / self.onnx_input_shape[1]
                rows = self.single_array.shape[2] / self.onnx_input_shape[2]
                original_size = self.single_array.shape[1:]
                self.onnx_batch = (int(cols) if int(cols) == cols else int(cols) + 1,
                                   int(rows) if int(rows) == rows else int(rows) + 1,)

                self.padding_size = (self.onnx_batch[0] * self.onnx_input_shape[1] - self.single_array.shape[1],
                                     self.onnx_batch[1] * self.onnx_input_shape[2] - self.single_array.shape[2],)

                self.single_array = np.pad(self.single_array[:3], ((0, 0),
                                                                   (0, self.padding_size[0]),
                                                                   (0, self.padding_size[1])),
                                           "constant"
                                           )
                print(self.single_array.shape)
                self.padding_size = original_size

        ########################################################################################################################
        except:
            QMessageBox.warning(self, "Unknown Error!", "发生未知错误：图像未导入，可能图像格式不支持！")
    def open_single_image(self):
        file_name = QFileDialog.getOpenFileName(self, "请选择图像：", '.', "Image files (*.bmp *.jpg *.png *.tiff *.tif)")[0]
        self.input_image(file_name)

    def process_single_image(self):
        print("test")
        # 调用onnx模型，推理输出图像
        try:
            if self.single_array is not None:
                # input_texture = texture_loader(self.single_array,
                #                                self.single_array.min(),
                #                                self.single_array.max(),
                #                                16,
                #                                3)

                input_image  = cv2.normalize(self.single_array, None,
                                             0.0, 1.0,
                                             cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                del self.single_array
                # input_texture = cv2.normalize(input_texture, None,
                #                               0.0, 1.0,
                #                               cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                input_image   = input_image  .reshape(1, input_image.shape[0],
                                                      input_image.shape[1],
                                                      input_image.shape[2])
                # input_texture = input_texture.reshape(1, input_texture.shape[0],
                #                                       input_texture.shape[1],
                #                                       input_texture.shape[2])
########################################################################################################################
                output = np.zeros([self.onnx_batch[0] * self.onnx_input_shape[1],
                                  self.onnx_batch[1] * self.onnx_input_shape[2]])
                for cols in range(self.onnx_batch[0]):
                    for rows in range(self.onnx_batch[1]):
                        each_batch_image = input_image[:,:,
                                                       cols * self.onnx_input_shape[1] : (cols+1) * self.onnx_input_shape[1],
                                                       rows * self.onnx_input_shape[2] : (rows+1) * self.onnx_input_shape[2]
                                                      ]
                        each_batch_image = self.sess.run([self.output_name], {self.input_image_name : each_batch_image})
                        each_batch_image = np.array(each_batch_image[0][0][0])
                        each_batch_image[each_batch_image >= 0.4] = 1
                        each_batch_image[each_batch_image <  0.4] = 0
                        output[cols * self.onnx_input_shape[1] : (cols+1) * self.onnx_input_shape[1],
                               rows * self.onnx_input_shape[2] : (rows+1) * self.onnx_input_shape[2]] = each_batch_image

                output = output[:self.padding_size[0], :self.padding_size[1]]
########################################################################################################################
                # self.single_array = self.sess.run([self.output_name], {self.input_image_name  : input_image  ,
                #                                                        self.input_texture_name: input_texture})
                # self.single_array = np.array(self.single_array[0][0][0])
                #
                # self.single_array[self.single_array >= 0.5] = 1
                # self.single_array[self.single_array <  0.5] = 0
########################################################################################################################
                print(self.padding_size)
                print(output.shape)
                changed_h, changed_w = self.return_resize(430, 430)
                pixmap = output
                pixmap = cv2.normalize(pixmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                pixmap = cv2.resize(pixmap, (changed_w, changed_h), interpolation=cv2.INTER_NEAREST)

                pixmap = QImage(pixmap, changed_w, changed_h, changed_w,QImage.Format_Grayscale8)
                pixmap = QPixmap(pixmap)
                pixmap = QGraphicsPixmapItem(pixmap)
                scene  = QGraphicsScene()
                scene   .addItem(pixmap)
                self.graphicsResview.setScene(scene)
                self.whether_finished = "Finished"
                self.single_array = output
            else:
                QMessageBox.information(self, "Process Blocked!", "进程阻滞：未导入图像！")

        except:
            QMessageBox.warning(self, "Unknown Error!", "发生未知错误：无法处理图像或图像未导入！")


    @staticmethod
    def image_writer(project, transform, image, output_path):
        driver = gdal.GetDriverByName("GTiff")
        out = driver.Create(output_path, image.shape[1], image.shape[0], gdal.GDT_Byte)
        out.SetGeoTransform(transform)
        out.SetProjection(project)
        out.GetRasterBand(1).WriteArray(image)
        out.FlushCache()
        out = None

    def save_image(self,image_path):
        if self.whether_finished == "Finished":
            try:
                self.single_array = cv2.normalize(self.single_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.image_writer(self.single_proj,
                                  self.single_trans,
                                  self.single_array,
                                  image_path)

            except:
                QMessageBox.warning(self, "Unknown Error!", "发生未知错误：无法保存图像！")
        else:
            QMessageBox.information(self, "Process Blocked!", "进程阻滞：没有可保存的图像！")
    def save_single_image(self):
        file_name = QFileDialog.getSaveFileName(self, "请选择保存路径：", '.', "TIF(*.TIF)")[0]
        self.save_image(file_name)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = UiMain(r'ImageClassifier.onnx')
    ui.show()
    sys.exit(app.exec())
