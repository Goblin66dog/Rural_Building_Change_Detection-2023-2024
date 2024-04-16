# Deep Learning
 深度学习算法总结（Personal Configuration）

## 目录：
- [项目の目的](#项目の目的)
- [项目の结构](#项目の结构)
- [项目の详述](#项目の详述)
  - [训练准备](#训练准备)
    - [数据读取](#数据读取) 
    - [数据处理](#数据处理)
    - [网络库](#网络库)
  - [训练配置](#训练配置)
    - [数据分配](#数据分配)
    - [损失函数](#损失函数)
    - [优化器](#优化器)
    - [训练策略](#训练策略)
  - [模型部署（后话）](#模型部署)
- [Thanks For Supporting!](#Thanks For Supporting!)

# 1. 项目の目的
# 2. 项目の结构
<pre> Datasets
->数据读取
->数据处理
->数据分配
->网络训练<-训练策略(损失函数+优化器)
->输出
</pre>

# 3. 项目の详述

## 3.1 数据读取
***
**_Data_Reader_**
- **数据读取模块**
- 用于各种栅格数据读取
  - 读取包括.TIF/.PNG/.JPG的各种图像
  - 返回的栅格类型为：C·H·W
- 代码更加精简
<pre>
- Dataset          : 定义了一个Dataset类
  - input_file_path: 输入文件路径
  - self.data      : gdal对象
  - self.width     : 图像宽
  - self.height    : 图像高
  - self.proj      : 地图投影信息
  - self.geotrans  : 仿射变换参数
  - self.array     : 图像数据
  - self.bands     : 图像波段（通道数）
  - self.type      : 图像数据格式
- 无返回值
</pre>
***
## 3.2 数据处理
**_Padding_**
- **对单张图像进行Padding操作**
<pre>
- Padding           : 定义了一个Padding类
  - image           : 输入图像栅格
  - image_shape     : N·C·H·W的顺序
  - self.mir        : 镜像Padding操作
    - target_height : Padding目标高 
    - target_width  : Padding目标宽
  - self.nor        : 常规Padding操作（直接补0）
    - target_height : Padding目标高 
    - target_width  : Padding目标宽
  - self.min        : 最小Padding操作（用于适应多层下采样或卷积）
    - divide        : 需要适应的被除数
    - 函数返回为输入图像与divide之间的最小公倍数大小的图像
- 返回一个经Padding操作的图像
</pre>
***
**_Random_Flip_**
- **对原始图像和标签图像进行随机反转操作**
<pre>
- RandomFlip        : 定义了一个RandomFlip类
  - image           : 输入原始图像栅格
  - label           : 输入标签图像栅格
  - random_flip     : 随机翻转
- 返回一对经随机翻转的图像元组
</pre>
***

***
**_Data_Loader_**
- **数据加载模块**
- **适应于单输入图像网络**
- **不进行任何数据预处理**
- 用于将样本进行读取
  - 读取后将栅格存储进入Dataloader中
  - 返回为栅格序列
  - 返回的栅格类型为：N·C·H·W


