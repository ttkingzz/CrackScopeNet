# CrackScopeNet

CrackScopeNet: An Efficient Crack Segmentation for Resource-Constrained Structural Health Monitoring

</hr>

# Abstract

Timely detection of cracks is a critical task in structural health monitoring. However, current neural network models often require substantial computational resources and intricate designs, which limits their practicality in time-sensitive, resource-constrained outdoor environments. To bridge this gap, we introduce CrackScopeNet, a lightweight segmentation network model designed based on common crack morphology. This model is suitable for deployment on devices with limited computational power and memory, such as drones, mobile robots, and edge devices. CrackScopeNet features a lightweight multi-scale feature extraction module to capture local contextual information of cracks and a stripe-wise context attention mechanism to obtain long-distance dependencies, mitigating the effects of poor lighting, shadows, and obstructions. We evaluate the model on the composite dataset CrackSeg9k, demonstrating its remarkable efficiency and outstanding performance. The model achieves superior mean intersection over union (mIoU) scores compared to state-of-the-art models while maintaining low parameters and computational overhead. Furthermore, CrackScopeNet shows ideal inference speed on edge devices with severe computational and memory constraints without GPU support. CrackScopeNet contributes to the development of efficient and effective crack segmentation networks for practical structural health monitoring applications in resource-limited environments.

# Model Architecture


# Datasets
[CrackSeg9k](https://github.com/Dhananjay42/crackseg9k)

[Ozgenel](https://data.mendeley.com/datasets/jwsn7tfbrp/1)

[Aerial Track Dataset](https://github.com/zhhongsh/UAV-Benchmark-Dataset-for-Highway-Crack-Segmentation)

We train the model on a comprehensive dataset (CrackSeg9k) and subsequently transfer to specific downstream scenarios, concrete crack (Ozgenel) and earthquake crack(Aerial Track Dataset).

# More
More details will be available after the article is accepted.