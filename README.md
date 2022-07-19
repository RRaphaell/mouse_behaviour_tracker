![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white) 
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) 
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rraphaell-mouse-behaviour-tracker-app-so205q.streamlitapp.com/)

## **Overview**

This project aims to automate the tracking of rats' behavior.

We use deep learning to track a rats position and analyze it based on that information. <br>
Using the canvas, you can draw segments and it generates a video marked the rats' bodies parts and also making some statistics, such as the time spent in each segment, number of segments crossings, and visualization of the entire path. (in the future, we might add some analysis)


https://user-images.githubusercontent.com/74569835/179845647-7773e224-a42a-4deb-9d32-17e46b8ac8da.mp4


## **How to use**
you can just open the [website](https://rraphaell-mouse-behaviour-tracker-app-so205q.streamlitapp.com/) and upload the video you want to track, or use our example if you just want to see how it works.<br>
Draw all segments on the canvas and click the start button to generate the report.<br>
Based on the length of the video, it may take several minutes

## **Features**
- easy. no installation needed
- friendly and sophisticated UI
- allows specific segment analysis
- modern reporting
- continuously developing
- open to contribute

## **Run locally**
### **Installation**
#### **Quick install**
```
pip install -r requiremenets.txt
```
#### **Using conda (recommended)**
```
conda create -n <env_name> python==3.8
conda activate <env_name>
pip install -r requiremenets.txt
```
### **how to run**
```
streamlit run app.py
```


## **Project structure graph**
Using this illustration, we can see how the project works in general.<br>
We have two related models, one for detecting the center of the rat and the remaining one for detecting rats body parts (nose, left eye, right eye, backbone, tail start, tail end). We use the first model to detect the backbone of rats, then crop the image around it and use the second model to detect body parts accurately. Preprocessing techniques such as rescaling, normalization, etc., are applied.

[![Screenshot-from-2022-07-01-16-55-53.png](https://i.postimg.cc/SNLbm4ng/Screenshot-from-2022-07-01-16-55-53.png)](https://postimg.cc/GHp5jZJD)

## **File structure**
this tree graph helps you to dive into our code and understand project file structure quickly.  

```bash
.
├── app.py    # main script
├── Dockerfile
├── environment.yml
├── README.md
├── requirements.txt
├── scripts   # all necessary scripts
│   ├── Models    # two models with training and inference scripts
│   │   ├── CenterDetector    # model for center detector
│   │   │   ├── center_detector_training.ipynb    # center detection model training notebook
│   │   │   ├── config.py       
│   │   │   ├── Dataset.py        
│   │   │   └── weights.bin 
│   │   ├── Controller    # module that controls both model predictions and processing
│   │   │   ├── config.py
│   │   │   ├── Controller.py
│   │   │   └── utils.py
│   │   ├── PartsDetector   # model for parts detections. using after the center detection model
│   │   │   ├── config.py
│   │   │   ├── Dataset.py
│   │   │   ├── parts_detector_training.ipynb_
│   │   │   └── weights.bin
│   │   ├── Dataset.py
│   │   ├── Metrics.py
│   │   ├── ModelBuilder.py
│   │   ├── Trainer.py
│   │   ├── UNet.py   # Unet model written from scratch
│   │   └── utils.py
│   ├── Report    # module to show all analyses based on rats' tracking
│   │   ├── Bar.py
│   │   ├── Card.py
│   │   ├── Dashboard.py
│   │   ├── Pie.py
│   │   └── utils.py
│   ├── Analyzer.py
│   ├── config.py
│   ├── Pipeline.py
│   ├── Tracker.py
│   └── utils.py
├── style.css
```


## Contact
if you have any questions, ideas please contact us <br> 

raffo.kalandadze@gmail.com
[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/raphael-kalandadze-ab9623142/)
[![Twitter](https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/RaphaelKalan)


tatiatsmindashvili@gmail.com
[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tatia-tsmindashvili-92676614b/) 
[![Twitter](https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/TatiaTsmindash1)


## **License**

