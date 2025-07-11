Deep adversarial data augmentation for biomedical spectroscopy: Application to modelling Raman spectra of bone,
Chemometrics and Intelligent Laboratory Systems,
Volume 228,
2022,
104634,
ISSN 0169-7439,
https://doi.org/10.1016/j.chemolab.2022.104634.
(https://www.sciencedirect.com/science/article/pii/S0169743922001459)
Abstract: Deep learning algorithms have performed remarkably well to predict state of health. Nevertheless, they typically rely on ample training data to avoid overfitting. In the biomedical sector, sufficient data are not typically available due to low availability or accessibility. Data augmentation of physiological recordings can be achieved using Generative Adversarial Networks (GAN). GAN is a computational framework for approximating generative models within an adversarial process, where two neural networks compete against one other while being trained simultaneously. Despite the widespread use and adoption of deep learning algorithms in life sciences, concerns have been raised about the lack of biological context. Therefore, to assess a data augmentation workflow, both computational and physiological quality metrics must be considered. Raman spectroscopy can be effectively used to study the molecular properties of bone tissue. Both inorganic and organic phases can be analysed simultaneously as probes of bone health status. In this work, we describe an easy-to-follow GAN approach for generating synthetic Raman spectra from a small dataset of ex vivo healthy and osteoporotic bone samples. The model was applied to raw Raman spectra, while it can be modified accordingly to produce any one-dimensional biomedical signal. We also introduced a novel unsupervised methodology to evaluate the variability of the synthetic dataset, based on successive Principal Component Analysis (PCA) modelling. The properties of the synthetic spectra were scrutinized by Fréchet Distance and difference spectroscopy, as well as by bone quality metrics, like mineral-to-matrix ratio and crystallinity. Finally, classification studies demonstrated the increased discrimination accuracy of the augmented dataset.
Keywords: Generative adversarial networks; Raman spectroscopy; Bone; Deep learning


# ganram
GAN-BASED DATA AUGMENTATION for RAMAN SPECTRA 

A generative adversarial network implementation in TensorFlow to perform data augmentation of biomedical Raman spectra.

## Requirements
tensorflow 2.6.0 (Requires: absl-py, protobuf, tensorflow-estimator, keras-preprocessing,astunparse, six, wheel, grpcio, flatbuffers, clang, h5py, tensorboard, gast, opt-einsum, termcolor, wrapt, keras, numpy, google-pasta, typing-extensions);  
keras 2.6.0;  
python 3.7.11;  
pandas 1.3.3;  
scipy 1.7.1;  
matplotlib 3.4.2 (Requires: kiwisolver, numpy, python-dateutil, pillow, pyparsing, cycler);  
openpyxl=3.0.9;  

For the Scavenging PCA code:  
Our libraries: dffuncs.py, modeling.py and preprocessing.py;  
numpy 1.19.2;  
seaborn 0.11.2;  
sklearn;


