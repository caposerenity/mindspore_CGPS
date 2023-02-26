
## CGPS描述

CGPS[论文地址参见](https://arxiv.org/pdf/2106.10506.pdf)是AAAI2022的中稿工作，论文全名为Exploring Visual Context for Weakly Supervised Person Search。该方法基于Faster-RCNN，是首篇弱监督设定下的行人搜索工作。

如下为MindSpore使用CUHK-SYSU数据集对DAPS进行训练的示例。

## 性能

|  Dataset  | Model |  mAP  | Rank1 |
| :-------: | :---: | :---: | :---: |
| CUHK-SYSU | CGPS  | 80.1% | 82.1% |
|    PRW    | CGPS  | 16.6% | 68.2% |

## 数据集

使用的数据集：[CUHK-SYSU](https://github.com/ShuangLI59/person_search)和[PRW](https://github.com/liangzheng06/PRW-baseline)

全部下载好后，我们提供了COCO格式的[标注文件](https://github.com/daodaofr/AlignPS/tree/master/demo/anno)

对于CUHK-SYSU数据集，可以在配置文件的第30，31行修改数据集和标注的地址

## 环境要求

  - 硬件
    - 准备Ascend处理器搭建硬件环境。
  - 框架
    - [MindSpore](https://www.mindspore.cn/install/en)，本模型编写时版本为r1.2，12.30更新由r1.5编写的版本。
  - 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 脚本及样例代码

```shell
.
└─project1_fasterrcnn
  ├─README.md                           
  ├─scripts
    └─run_eval_ascend.sh                
    └─run_eval_gpu.sh                   
    └─run_eval_cpu.sh                   
  ├─src
    ├─FasterRcnn
      ├─label_generators                // 使用聚类（DBSCAN）生产id伪标签
      ├─__init__.py                     
      ├─cgps_head.py                    // 检测和重识别头
      ├─anchor_generator.py             // 生成anchor
      ├─faster_rcnn_r50.py              // 模型定义
      ├─fpn_neck.py                     // neck层
      ├─resnet50.py                     // ResNet-50
      ├─roi_align.py                    // ROI Align层
      └─rpn.py                          // Region Proposal Network
    ├─config.py               
    ├─dataset.py              
    ├─lr_schedule.py          
    ├─network_define.py       
    └─util.py                 
  ├─cocoapi                   
  ├─pretrained_faster_rcnn.ckpt         
  ├─eval.py                   // evaluation script
  └─train.py                  // training script
```

## 环境准备
```shell
pip install -r requirements.txt

# install COCO evaluation API
cd cocoapi/PythonAPI
python setup.py install
```



