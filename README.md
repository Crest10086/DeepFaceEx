# DeepFaceEx
DeepFaceEx是一个与DeepFaceLab配合使用的插件

## 安装

所有文件复制到DeepFaceLab目录下即可，如下图所示：


## 根据人脸相似度排序

在多个人脸的视频中，挑选出想要提取的人脸，例如：我们想快速提取出复仇者联盟中，黑寡妇的照片

1.提取复仇者联盟里所有人脸，大概4万多张人脸
![](https://github.com/dotapuppy/DeepFaceEx/blob/master/Images/org.png)

2.选择一张黑寡妇的照片
![](https://github.com/dotapuppy/DeepFaceEx/blob/master/Images/select.png)

3.排序后，和所选照片越相似的人脸，会排在越前面。由于要考虑到运行速度，人脸特征值的算法并不是很强，所以还是有一些误检存在
![](https://github.com/dotapuppy/DeepFaceEx/blob/master/Images/sorted.png)

4.往下拖动，当大部分人脸都不是黑寡妇时，删除所有图片

5.再用DeepFaceLab本身的sort by similar histogram功能，快速删除其他人的照片
