## Removing input features via a generative model to explain their attributions to classifier's decisions

This repository contains source code necessary to reproduce some of the main results in the paper:


**If you use this software in an academic article, please consider citing:**
    
## 1. Setup

### Installing software
This code is built using PyTorch. You can install the necessary libraries by pip installing the requirements text file using the command: **pip install -r ./requirements.txt**

## 2. Usage
The main codes for SP, LIME, and MP are in [formal_SP_single_image.py](formal_SP_single_image.py), [formal_LIME_single_image.py](formal_LIME_single_image.py), and [formal_MP_single_image.py](formal_MP_single_image.py). In addition, after installing the LIME library you will have to replace the *lime_image.py* script with our [formal_lime_image.py](formal_lime_image.py). Three shell scripts have been provided which for a given an [image](teaser_image.JPEG) and target class generates its respective attribution maps for an algorithm and its generative version.

## 4. Licenses
Note that the code in this repository is licensed under MIT License, but, the pre-trained condition models used by the code have their own licenses. Please carefully check them before use. 

## 5. Questions?
If you have questions/suggestions, please feel free to [email](mailto:chiragagarwall12@gmail.com) or create github issues.
