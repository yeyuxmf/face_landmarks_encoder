# face_key_points
Face landmarks, key points, CNN, Transformer, Pytorch, 300W\ WFLW dataset    
This is a direct coordinate regression method that eliminates the post-processing required by the heatmap-based approach.
# Model introduction
"The paper is currently being prepared." Ongoing updates  
This is a regression-based model, which has the characteristics of high accuracy, no missing points and low memory consumption compared with the best heat map model in terms of accuracy.

# DataSet
Data preprocessing comes from: 
1. https://github.com/huangyangyu/ADNet
2. https://github.com/ZhenglinZhou/STAR

# Model result # model The extraction code is: 1234
batch size =1, GPU: RTX 3060,  CPU: 12th Gen Intel Core(TM) i7-12700F 2.1GHz
Dataset | Model | test gpu| gflops G | params M | FPS  | ION | IPN | model weights 
--- | --- | --- | --- | --- | --- | --- | --- | --- 
WFLW | ResNet34    + encoder | 418MB | 11.80 | 53.52 | 70.7  | 4.0065 | 5.6676 | [baidu](https://pan.baidu.com/s/1_eJ-h2f8McT4FLbvYOblZw)
WFLW | ResNet18    + encoder | 269MB | 5.246 | 20.88 | 149.5 | 4.0513 | 5.7311 | [baidu](https://pan.baidu.com/s/1OXZunG99sPmfzkh_wWPdlg)
WFLW | MobileNetV3 + encoder | 195MB | 1.187 | 6.063 | 198.5 | 4.1837 | 5.9193 | [baidu](https://pan.baidu.com/s/1sRGdWvxnCBm6a_ETL9o2vA)
WFLW | MobileNetV3 + encoder | 194MB | 0.720 | 3.908 | 236.7 | 4.2798 | 6.0575 | [baidu](https://pan.baidu.com/s/1cxT5pok8B3p14fVLbmYCLw)
WFLW | MobileNetV3 + encoder | 210MB | 0.603 | 3.442 | 284.3 | 4.4047 | 6.3332 | [baidu](https://pan.baidu.com/s/12eBRv9EnsFYGSWK06BAxqw)
300W | ResNet34    + encoder | 257MB | 10.20 | 44.52 | 60.5  | 2.8677 | 4.0238 | [baidu]()
300W | ResNet18    + encoder | 257MB | 5.117 | 20.88 | 153.5 | 2.8797 | 4.0409 | [baidu](https://pan.baidu.com/s/1NAxkBTTMxx4meAk2Ao54pw)
300W | MobileNetV3 + encoder | 220MB | 1.861 | 9.922 | 157.5 | 2.9365 | 4.1209 | [baidu](https://pan.baidu.com/s/1vYoox7kgyh9rY2RF4IUavg)
300W | MobileNetV3 + encoder | 180MB | 0.774 | 3.897 | 249.8 | 2.9403 | 4.1255 | [baidu](https://pan.baidu.com/s/1qey_OruuDY17mo97n5Nhk)
300W | MobileNetV3 + encoder | 170MB | 0.707 | 3.700 | 298.1 | 3.1742 | 4.4541 | [baidu](https://pan.baidu.com/s/1PakB77oi4r0LAHKuIsQluA)
300W | MobileNetV3 + encoder | 170MB | 0.601 | 3.442 | 286.1 | 3.0328 | 4.2559 | [baidu](https://pan.baidu.com/s/1AGtCWIn2nU6xX7nOcwRUwQ)
300W | Comparative：  STAR   | 2400MB| 17.05 | 13.37 | 55.1  | 2.8704 | ~~~~~~ | ~~~~~


STAR：The STAR results were obtained through testing using the authors' original code and model weight files.
It currently achieves the best results on the 300W and WFLW face datasets, belonging to the heatmap-based landmark model.https://github.com/ZhenglinZhou/STAR



# Reference paper:
1. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.   
2. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16000-16009).   
3. 苏剑林. (Sep. 22, 2025). 《重新思考学习率与Batch Size（四）：EMA 》[Blog post]. Retrieved from https://kexue.fm/archives/11301.   
4. 苏剑林. (May. 10, 2021). 《Transformer升级之路：4、二维位置的旋转式位置编码 》[Blog post]. Retrieved from https://kexue.fm/archives/8397.   
5. Dai C, Wang Y, Huang C, et al. A Cephalometric Landmark Regression Method Based on Dual-Encoder for High-Resolution X-Ray Image[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 93-109.   
6. He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.   
7. Ultralytics. (2024). ultralytics/ultralytics: A unified deep learning framework for computer vision tasks (v8.2.0) [Computer software]. GitHub. https://github.com/ultralytics/ultralytics.    


These are the model results for November 2023. I haven't conducted further experiments or written a paper yet, but I plan to do so by the end of this year.
The source code will be fully open by then.





