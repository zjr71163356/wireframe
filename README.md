## Code of paper "Learning to Parse Wireframes in Images of Man-Made Environments", CVPR 2018

| Folder/file       | Description                  |
|------------|------------------------------|
| junc      | For training junction detector. |
| linepx    | For training straight line pixel detector. |
| wireframe.py | Generate line segments/wireframe from predicted junctions and line pixels. |
| evaluation | Evaluation of junctions and wireframes. |
测试
## Requirements
- python3
- pytorch==0.3.1
- opencv==3.3.1 
- scipy, numpy, progress, protobuf
- joblib (for parallel processing data.)
- tqdm
- [optional] dominate

The code is written and tested in `python3`, please install all requirements in python3.

## Prepare data
- Download the training data.
    - Imgs, train annotaions, test annotations are all available at [BaiduPan](https://pan.baidu.com/s/11CKr5s0zHnuVKsJVXianxA?pwd=wf18) (passwd: wf18).
    - You can also download data from onedrive: [onedrive](https://1drv.ms/u/s!AqQBtmo8Qg_9uHpjzIybaIfyJ-Zf?e=Fofbch)(no passwd required).
 
- put training data in __data/__ and test annotation in put it in __evaluation/wireframe/__,
```shell
    unzip v1.1.zip
    unzip pointlines.zip
    unzip linemat.zip
```

    
- Data Structure  
    Each .pkl file contains the annotated wireframe of an image, and it consists of the following variables:  
    ```shell
    *.pkl  
        |-- imagename: 	the name of the image  
        |-- img:         the image data  
        |-- points:      the set of points in the wireframe, each point is represented by its (x,y)-coordinates in the image  
        |-- lines:       the set of lines in the wireframe, each line is represented by the indices of its two end-points  
        |-- pointlines:     the set of associated lines of each point        
        |-- pointlines_index:       line indexes of lines in 'pointlines'  
        |-- junction:       the junction locations, derived from the 'points' and 'lines'  
        |-- theta:      the angle values of branches of each junction                   
    ```
- visualizing the wireframe.  
  After loading the .pkl file, you can run something like the following in Python to visualize the wireframe:
  ```python
    for idx, (i, j) in enumerate(lines, start=0):
        x1, y1 = points[i]
        x2, y2 = points[j]
        cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_8)
  ```

- Preprocess data.
    ```
    cd junc
    python3 main.py --create_dataset --exp 1 --json

    cd linepx
    python3 main.py --genLine
    ```
Note: `--json` means you put the hype-parameters in __junc/hypes/1.json__.

## Training
- train junction detector.
    ```
    cd junc
    python3 main.py --exp 1 --json --gpu 0 --balance
    ```

- train line pixel detecor.
    ```
    cd linepx
    python3 main.py --netType stackedHGB --GPUs 0 --LR 0.001 --batchSize 4
    ```

## Testing
- Test junction detector.
    ```
    cd junc
    python3 main.py --exp 1 --json --test --checkepoch 16 --gpu 0 --balance
    ```
- Test line pixel detector.
    ```
    cd linepx
    python3 main.py --netType stackedHGB --GPUs 0 --LR 0.001 --testOnly t
    ```
- Combine junction and line pixel prediction.
    ```
    python wireframe.py
    ```

### Evaluation
The code for evaluation is put in [evaluation/junc](evaluation/junc) and [evaluation/wireframe](evaluation/wireframe).
Expected junction and wireframe precision/recall curve is like
<figure class="half">
    <img src="evaluation/junc/junc_1_16.png", width=400/> 
</figure>

<figure class="half">
    <img src="evaluation/wireframe/1_0.5_0.5.png", width=400/>
</figure>


### Visualize the result
For visualizing the result, we recommend generating an html file using [dominate](https://github.com/Knio/dominate) to
visualize the result of different methods in columns.


## Citation
```
@InProceedings{wireframe_cvpr18,
author = {Kun Huang and Yifan Wang and Zihan Zhou and Tianjiao Ding and Shenghua Gao and Yi Ma},
title = {Learning to Parse Wireframes in Images of Man-Made Environments},
booktitle = {CVPR},
month = {June},
year = {2018}
}
```

## License
You can use this code/dataset for your research and other usages, following MIT License.




- `--create_dataset`: 创建数据集。
- `--json`: 从文件加载架构参数。
- `--balance`: 平衡正负样本的比例。
- `--ratio`: 正负样本的比例，默认为7。
- `--trainer`: 指定训练器名称，默认为`balance_junction`。
- `--criterion`: 指定损失函数名称，默认为`balance`。
- `--decoder`: 指定解码器名称，默认为`junction`。
- `--split`: 指定数据集划分，默认为`train`。
- `-e`, `--exp`: 实验名称，默认为`1`。
- `--net`: 指定特征网络，默认为`inception`。
- `--test`: 是否进行测试。
- `--image_size`: 输入图像尺寸，默认为480。
- `--grid_size`: 输出尺寸，默认为60。
- `--focus_size`: 每个网格单元覆盖的范围，默认为1.5。
- `--num_bin`: 0到360的分箱数量，默认为15。
- `--max_len`: 每个单元格的最大连接数，默认为1。
- `--loss_weights`: 连接和分箱损失的权重，默认为`1.0, 0.1, 1.0, 0.1`。
- `--decodeFeats`: 解码器网络的通道数，默认为256。
- `--valIntervals`: 验证间隔，默认为5。
- `--gpu`: 指定使用的GPU设备，默认为`0`。
- `--epochs`: 训练的轮数，默认为17。
- `--save_dir`: 保存模型的目录，默认为`output`。
- `--num_workers`: 加载数据的工作线程数，默认为1。
- `--batch_size`, `-b`: 批处理大小，默认为1。
- `-o`, `--optimizer`: 训练优化器，默认为`sgd`。
- `--lr`: 初始学习率，默认为0.01。
- `--lr_steps`: 学习率调整的步数，默认为`8,12,16`。
- `--lr_decay_step`: 学习率衰减的步数，默认为5。
- `--lr_decay_gamma`: 学习率衰减比率，默认为0.1。
- `--clip_norm`: 优化时的梯度裁剪阈值，默认为1.0。
- `-r`, `--resume`: 是否恢复检查点，默认为False。
- `--checksession`: 加载模型的会话，默认为1。
- `--checkepoch`: 加载模型的轮数，默认为16。
- `--checkdir`: 保存检查点的路径，默认为空字符串。
- `--checkpoint`: 加载模型的检查点，默认为0。