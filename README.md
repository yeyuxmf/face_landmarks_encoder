# face_key_points
Face landmarks, key points, CNN, Transformer, Pytorch, 300W\ WFLW dataset    
This is a direct coordinate regression method that eliminates the post-processing required by the heatmap-based approach.
# Model introduction
This is a regression-based model, which has the characteristics of high accuracy, no missing points and low memory consumption compared with the best heat map model in terms of accuracy.
It is possible to modify the model structure to reduce parameters, speed up inference, and enable deployment on edge devices.
# DataSet
Data preprocessing comes from: 
1. https://github.com/huangyangyu/ADNet
2. https://github.com/ZhenglinZhou/STAR


# Train
#parser.add_argument('--data_definition', type=str, default='WFLW', help="COFW, 300W, WFLW")

python  train_300w.py    

# Test

python  test_jit_model.py


# Paper Model result 
paper_model_weights:

batch size =1, GPU: RTX 5070ti,  CPU: Intel(R) Core(TM) i9-14900KF   3.20 GHz

Dataset | Model | test gpu| gflops G | params M | FPS  | ION |  model weights |提取码: 
--- | --- | --- | --- | --- | --- | ---  | --- | --- 
WFLW | ResNet34    + encoder | 535MB | 11.80 | 53.49 | 81.1  | 3.97 | [baidu](https://pan.baidu.com/s/1WSM5m_uH0FsT4TnJLVhsBg?pwd=mw6a)| mw6a
WFLW | ResNet18    + encoder | 493MB | 9.38  | 43.38 | 86.7  | 4.01 | [baidu](https://pan.baidu.com/s/1EwwmdCNkWsGsM7EwjGNwlA?pwd=n2yg)| n2yg
WFLW | ShuffleNetV2 + encoder| 351MB | 2.58  | 9.49  | 131.1 | 4.11 | [baidu](https://pan.baidu.com/s/18MZLUBYs2aY3XJMgsbR-mQ?pwd=8xre)| 8xre
300W | ResNet34    + encoder | 503MB | 11.04 | 53.49 | 82.8  | 2.85 | [baidu](https://pan.baidu.com/s/1RH_zqWhzegqkfqQSCDcUtQ?pwd=shkr)| shkr
300W | ResNet18    + encoder | 461MB | 8.62  | 43.38 | 88.2  | 2.88 | [baidu](https://pan.baidu.com/s/1KOnpJkhKIeb4C7YT7t5Z0Q?pwd=3n4g)| 3n4g
300W | ShuffleNetV2 + encoder| 319MB | 2.45  | 9.49  | 137.1 | 2.91 | [baidu](https://pan.baidu.com/s/17uqemoi6OYGVm_BgvAhIiQ?pwd=75su)| 75su







