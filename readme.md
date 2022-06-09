# 关于文件`m_img_match.py`
## 图像校正
- 对于红外与可见光图像首先需要知道拍摄的相机参数
- 详细的参数可参考[DJI相机参数](https://forum.dji.com/thread-225870-1-1.html)
- 以Zenmuse XT2 (640X512)举例，文件中需要得知的参数分别是：
```angular2html
传感器宽度Sensor Size Width: 10.88mm
传感器高度Sensor Size Height: 8.7mm
```
- 对应代码
```python
IR_SENSOR_WIDTH = 10.88
IR_SENSOR_HEIGHT = 8.7
VIS_SENSOR_WIDTH = 7.4
VIS_SENSOR_HEIGHT = 5.55
```
- 根据传感器尺寸求出图像单位像素的实际尺寸大小`DIR`、`DVIS`
```python
    # 获取红外与可见光的XMP信息
    ir_xmp = get_xmp_info(irimg_path)
    vis_xmp = get_xmp_info(visimg_path)

    # 计算图像单位像素的实际尺寸大小
    d_ir = ir_xmp['ImageWidth'] / IR_SENSOR_WIDTH
    d_vis = vis_xmp['ImageWidth'] / VIS_SENSOR_WIDTH
```
- 利用`pyexiv2`库读取图片中的XMP信息，封装到`m_img_match.py`，该文件返回一个字典包括图像的经纬度、高度、偏航角、俯仰角、翻滚角、焦距、图像尺寸大小
- 利用专利1的内容对图像进行几何校正
```python
    #几何校正
    ir_img = Geo_Correcition(ir_img,d_ir,ir_xmp['FocalLength'],ir_xmp['Flight-Yaw'])
    vis_img = Geo_Correcition(vis_img,d_vis,vis_xmp['FocalLength'],vis_xmp['Flight-Yaw'])
```
- 利用软件获取的参数进行桶型校正
```python
#桶型校正
    ir_img = undistort(ir_img, d_ir, ir_xmp['FocalLength'], 1)
    vis_img = undistort(vis_img, d_vis, vis_xmp['FocalLength'], 2)
```
- 关于`cv2.undistort(img,K,D)`函数：
```python
    #桶型校正
    cv2.undistort(img,K,D)
    第一个参数src，输入参数，代表畸变的原始图像；
    第二个参数cameraMatrix，为之前求得的相机的内参矩阵；
    第三个参数distCoeffs，为之前求得的相机畸变矩阵；
```
- 进行QPSO优化匹配参数
- 最后返回**最优化配准参数pa**以及**融合图fine_MATCH**
