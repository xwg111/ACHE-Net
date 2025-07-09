# ACHE-Net

Medical image segmentation is a pivotal technology in diagnostics and treatment planning, enabling precise identification and segmentation of biological structures. This study introduces ACHE-Net, a novel U-shaped network integrating Adaptive Decomposed Convolution (ADConv) and a Dynamic High-Frequency Feature Enhancement module (DHF). ADConv balances model performance and computational overhead by dynamically adjusting the convolution kernel size based on feature channel numbers. The DHF module enhances high-frequency components using dynamic Discrete Wavelet Transform (DWT), adjustable according to input feature resolution. Experimental results on five datasets (Glas, BUSI, Kvasir, CVC-ClinicDB, and ISIC2018) demonstrate that ACHE-Net outperforms other popular models with fewer parameters, achieving significant improvements in Dice scores and HD95 metrics. Source code is available at: https://github.com/xwg111/ACHENet.


## Experiment
In the experimental section, five publicly available and widely utilized datasets are employed for testing purposes. These datasets are:<br> 
GlaS (gland, with 165 images)<br>
ISIC-2018 (dermoscopy, with 2,594 images)<br>
Kvasir-SEG (endoscopy, with 1,000 images)<br> 
BUSI (breast ultrasound, with 647 images)<br> 
CVC-ClinicDB (colonoscopy, with 612 images)<br>  


In GlaS dataset, we split the dataset into a training set of 85 images and a test set of 80 images. <br>
In ISIC 2018 dataset, we adopt the official split configuration, consisting of a training set with 2,594 images, a validation set with 100 images, and a test set with 1,000 images. <br>
For other dataset, the images are randomly split into training, validation, and test sets with a ratio of 6:2:2.<br>
The dataset path may look like:
```bash
/Your Dataset Path/
├── BUSI/
    ├── Train_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Val_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Test_Folder/
        ├── img
        ├── labelcol
```


## Usage

---

### **Installation**
```bash
git clone https://github.com/xwg111/ACHENet.git
conda create -n env_name python=3.7
conda activate env_name
conda install pytorch==1.9.0 torchvision==0.14.1 torchaudio==0.10.0 -c pytorch -c nvidia
``` 


### **Training**
```bash
python train_model.py
```
To run on different setting or different datasets, please modify Config.py .


### **Evaluation**
```bash
python test_model.py
``` 


## Citation

Our repo is useful for your research, please consider citing our article. <br>
This article has been submitted for peer-review in the journal called *The visual computer*.<br>
```bibtex
@ARTICLE{ACHE-Net,
  author  = {Wenguang Xu, Qian Dong, et al},
  journal = {The Visual Computer}
  title   = {Enhancing Medical Image Segmentation with Adaptive Convolution and Dynamic High-Frequency Feature Enhancement},
  year    = {2025}
}
```


## Contact
For technical questions, please contact xwg987654@gmail.com .
