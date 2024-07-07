# CrackScopeNet

CrackScopeNet: A Lightweight Neural Network for Rapid Cracks Detection on Resource-Constrained Drone Platforms

</hr>

# Abstract

Detecting cracks in structural health monitoring is crucial for ensuring infrastructure safety and longevity. Using drones to obtain crack images and automate processing can improve the efficiency of crack detection. To address the challenges posed by the limited computing resources of edge devices in practical applications, we propose CrackScopeNet, a lightweight segmentation network model that simultaneously considers local and global crack features, suitable for deployment on drone platforms with limited computational power and memory. CrackScopeNet features a multi-scale branch to improve sensitivity to cracks of varying sizes without substantial computational overhead and a stripe-wise context attention mechanism to enhance the capture of long-range contextual information, mitigating the interference from complex backgrounds. Experimental results on CrackSeg9k dataset demonstrate that CrackScopeNet leads to a significant improvement in prediction performance, with the highest mean intersection over union (mIoU) scores reaching 82.12\%, while maintaining a lightweight architecture with only 1.05M parameters and 1.58G floating point operations (FLOPs). Besides, CrackScopeNet excels in inference speed on edge devices without GPU because of its low FLOPs. CrackScopeNet contributes to the development of efficient and effective crack segmentation networks, suitable for practical structural health monitoring applications using drone platforms.


# Model Architecture


# Datasets
[CrackSeg9k](https://github.com/Dhananjay42/crackseg9k)

[Ozgenel](https://data.mendeley.com/datasets/jwsn7tfbrp/1)

[Aerial Track Dataset](https://github.com/zhhongsh/UAV-Benchmark-Dataset-for-Highway-Crack-Segmentation)

We train the model on a comprehensive dataset (CrackSeg9k) and subsequently transfer to specific downstream scenarios, concrete crack (Ozgenel) and earthquake crack(Aerial Track Dataset).

# More
More details will be available after the article is accepted.