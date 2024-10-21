# DeepLearning-GPU-Project-Accelearted
This project is focused on utilizing GPU acceleration to optimize deep learning model training. It includes Python code designed to improve the efficiency and performance of model training, making it faster for complex computations on GPUs. The goal is to handle large datasets and achieve high-performance results in deep learning tasks by adjusting and optimizing the model architecture and training processes for a GPU environment.

Project components:
1. **Input Preprocessing Module** (`input_1_gpu.py`): Processes and converts data into a format suitable for model training, with a focus on efficient data loading and handling in a GPU environment.
2. **Model Construction Module** (`model_1_gpu.py`): Defines the deep learning model architecture, optimized for GPUs to ensure better performance in computationally intensive tasks.
3. **Training Module** (`train_1_gpu.py`): Manages the training process, implementing accelerated training strategies to ensure efficient execution on GPUs, reducing overall training time.

This project is ideal for developers and researchers working with large datasets who require GPU-based acceleration to improve the training speed and performance of deep learning models.


as for it's application
This project is primarily focused on underwater signal modulation classification using CNN and recurrent neural networks. It leverages multithreading to accelerate the training process. The training data should be placed in the same directory, containing labels, SNR values, features (IQ vectors), and IDs. Additionally, a CPU version of the model is provided for environments without GPU access.

Key components:
1. **Underwater Signal Modulation Classification**: Uses CNN and RNN to classify various underwater signal modulations.
2. **Multithreading for Faster Training**: The project employs multithreading techniques to speed up training, especially when large datasets are involved.
3. **Training Data Structure**: The training dataset includes essential fields like `label`, `SNR`, `feature (IQ vector)`, and `ID`, which need to be stored in the same directory for the model to function correctly.
4. **CPU Version**: In addition to the GPU-optimized version, a CPU-compatible version is available for users who may not have access to GPUs. 

This setup is designed to be flexible and efficient, making it suitable for signal processing and classification tasks in both research and practical applications.
