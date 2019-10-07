## Removing input features via a generative model to explain their attributions to classifier's decisions

This repository contains source code necessary to reproduce some of the main results in the paper:


**If you use this software in an academic article, please consider citing:**
    
## 1. Setup

### Installing software
This code is built using PyTorch. You can install the necessary libraries by pip installing the requirements text file using the command: **pip install -r ./requirements.txt**

## 2. Usage
The main codes for SP, LIME, and MP are in [formal_SP_single_image.py](formal_SP_single_image.py), [formal_LIME_single_image.py](formal_LIME_single_image.py), and [formal_MP_single_image.py](formal_MP_single_image.py). In addition, after installing the LIME library you will have to replace the *lime_image.py* script with our [formal_lime_image.py](formal_lime_image.py).
