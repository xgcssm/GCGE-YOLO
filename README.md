# 一款面向改进Yolov5的开源仓库，提供丰富的魔改方法
# An Open Source Repository for Improving Yolov5, Providing Rich Magic Methods


![image](https://user-images.githubusercontent.com/58406737/202331524-d57da9d4-bc93-492a-be05-4f5f4dcc9aab.png)


<div align="center">       <img src="https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-750560739-blue"
                        alt="QQ交流群">
                    <b><a href="https://github.com/WangQvQ/Yolov5_Magic">
                            <img src="https://img.shields.io/badge/%E8%BF%AA%E8%8F%B2%E8%B5%AB%E5%B0%94%E6%9B%BC-YOLO%20Magic-critical"
                                alt="迪菲赫尔曼">
                        </a>
</center></div>



-----

## 改进方式教程
## Tutorial on How to Improve


1. [手把手带你调参Yolo v5 (v6.2)（推理）](https://blog.csdn.net/weixin_43694096/article/details/124378167)🌟强烈推荐

2. [手把手带你调参Yolo v5 (v6.2)（训练）](https://blog.csdn.net/weixin_43694096/article/details/124411509?spm=1001.2014.3001.5502)🚀


3. [手把手带你调参Yolo v5 (v6.2)（验证）](https://yolov5.blog.csdn.net/article/details/126630836)


4. [如何快速使用自己的数据集训练Yolov5模型](https://blog.csdn.net/weixin_43694096/article/details/124457787)

5. [手把手带你Yolov5 (v6.2)添加注意力机制(一)（并附上30多种顶会Attention原理图）](https://blog.csdn.net/weixin_43694096/article/details/124443059?spm=1001.2014.3001.5502)🌟强烈推荐🍀新增8种

6. [手把手带你Yolov5 (v6.2)添加注意力机制(二)（在C3模块中加入注意力机制）](https://blog.csdn.net/weixin_43694096/article/details/124695537)

7. [Yolov5如何更换激活函数？](https://blog.csdn.net/weixin_43694096/article/details/124413941?spm=1001.2014.3001.5502)

8. [Yolov5如何更换BiFPN? ](https://yolov5.blog.csdn.net/article/details/125148552)

9. [Yolov5 (v6.2)数据增强方式解析](https://blog.csdn.net/weixin_43694096/article/details/124741952?spm=1001.2014.3001.5502)

10. [Yolov5更换上采样方式( 最近邻 / 双线性 / 双立方 / 三线性 / 转置卷积)](https://blog.csdn.net/weixin_43694096/article/details/125416120)

11. [Yolov5如何更换EIOU / alpha IOU / SIoU？](https://blog.csdn.net/weixin_43694096/article/details/124902685)

12. [Yolov5更换主干网络之《旷视轻量化卷积神经网络ShuffleNetv2》](https://blog.csdn.net/weixin_43694096/article/details/126109839?spm=1001.2014.3001.5501)

13. [YOLOv5应用轻量级通用上采样算子CARAFE](https://blog.csdn.net/weixin_43694096/article/details/126148795)

14. [空间金字塔池化改进 SPP / SPPF / SimSPPF / ASPP / RFB / SPPCSPC / SPPFCSPC](https://blog.csdn.net/weixin_43694096/article/details/126354660?spm=1001.2014.3001.5502)🚀

15. [用于低分辨率图像和小物体的模块SPD-Conv](https://blog.csdn.net/weixin_43694096/article/details/126398068)

16. [GSConv+Slim-neck 减轻模型的复杂度同时提升精度](https://blog.csdn.net/weixin_43694096/article/details/127344636?spm=1001.2014.3001.5501)🍀

17. [头部解耦 | 将YOLOX解耦头添加到YOLOv5 | 涨点杀器](https://yolov5.blog.csdn.net/article/details/127427578)🍀

18. [Stand-Alone Self-Attention | 搭建纯注意力FPN+PAN结构](https://yolov5.blog.csdn.net/article/details/127456629?spm=1001.2014.3001.5502)🍀
19. [YOLOv5模型剪枝实战](https://yolov5.blog.csdn.net/article/details/127576130)🚀
20. [YOLOv5知识蒸馏实战](https://yolov5.blog.csdn.net/article/details/127576321)🚀
21. [YOLOv7知识蒸馏实战](https://yolov5.blog.csdn.net/article/details/127943886)🚀
22. [改进YOLOv5 | 引入密集连接卷积网络DenseNet思想 | 搭建密集连接模块](https://blog.csdn.net/weixin_43694096/article/details/128500041?spm=1001.2014.3001.5501)🍀
23. [YOLOv5更换骨干网络之 PP-LCNet](https://yolov5.blog.csdn.net/article/details/128523175)🍀
24. [YOLOv5更换骨干网络之 EfficientNet-B0](https://yolov5.blog.csdn.net/article/details/128522888)🍀
25. [YOLOv5更换骨干网络之 MobileNet V3](https://yolov5.blog.csdn.net/article/details/128522041)🍀
26. [YOLOv5更换骨干网络之 GhostNet](https://yolov5.blog.csdn.net/article/details/128523623)🍀
------



## Performance
| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) | Weights
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------|------------------------------|-------------------------------|--------------------------------|--------------------|------------------------|------------------------|
| YOLOv5n                   | 640                   | 28.0                    | 45.7               | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                | [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt)
| YOLOv5s                   | 640                   | 37.4                    | 56.8               | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   | [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt)
| YOLOv5m                   | 640                   | 45.4                    | 64.1               | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   | [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt)
| YOLOv5l                   | 640                   | 49.0                    | 67.3               | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  | [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l.pt)
| YOLOv5x                   | 640                   | 50.7                    | 68.9               | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  | [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x.pt)
|                                                                                                      |                       |                         |                    |                              |                               |                                |                    |                        |
| YOLOv5n6                 | 1280                  | 36.0                    | 54.4               | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |[YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n6.pt)
| YOLOv5s6                 | 1280                  | 44.8                    | 63.7               | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |[YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt)
| YOLOv5m6                 | 1280                  | 51.3                    | 69.3               | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |[YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m6.pt)
| YOLOv5l6                 | 1280                  | 53.7                    | 71.3               | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |[YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l6.pt)
| YOLOv5x6<br>+ TTA | 1280<br>1536          | 55.0<br>**55.8**        | 72.7<br>**72.7**   | 3136<br>-                    | 26.2<br>-                     | 19.4<br>-                      | 140.7<br>-         | 209.8<br>-             |[YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x6.pt)

              
 <details><summary> <b>SPP Structure Parameter and GFLOPs</b> </summary>
 
| Model         | 参数量(parameters) | 计算量(GFLOPs) |
| ------------- | ------------------ | -------------- |
| SPP           | 7225885            | 16.5           |
| SPPF          | 7235389            | 16.5           |
| SimSPPF       | 7235389            | 16.5           |
| ASPP          | 15485725           | 23.1           |
| BasicRFB      | 7895421            | 17.1           |
| SPPCSPC       | 13663549           | 21.7           |
| SPPCSPC_group | 8355133            | 17.4           |

</details>

<details><summary> <b>Others Structure Parameter and GFLOPs</b> </summary>

| Model         | 参数量(parameters) | 计算量(GFLOPs) |
| ------------- | ------------------ | -------------- |
| TransposeConv upsampling| 7241917            | 16.6           |
| InceptionConv | 7233597            | 16.2           |
| BiFPN         | 7384006            | 17.2           |
| ShuffleNetv2  | 3844193            | 8.1            |
| CARAFE        | 7369445            | 17.0           |
</details>




<details><summary> <b>Update log</b> </summary>
2022.8.22 yolo.py Add Chinese annotations🍀

2022.8.24 Add Demo of Pyqt page🍀
</details>

<details><summary> <b>Acknowledgements</b> </summary>
https://github.com/ultralytics/yolov5
</details>
