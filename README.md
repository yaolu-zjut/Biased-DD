# Biased-DD
A repository of [Exploring the Impact of Dataset Bias on Dataset Distillation](https://arxiv.org/pdf/2403.16028.pdf)

## Abstract
Dataset Distillation (DD) is a promising technique to synthesize a smaller dataset that preserves essential information from the original dataset. This synthetic dataset can serve as a substitute for the original large-scale one, and help alleviate the training workload. However, current DD methods typically operate under the assumption that the dataset is unbiased, overlooking potential bias issues within the dataset itself. To fill in this blank, we systematically investigate the influence of dataset bias on DD. To the best of our knowledge, this is the first exploration in the DD domain. Given that there are no suitable biased datasets for DD, we first construct two biased datasets, CMNIST-DD and CCIFAR10-DD, to establish a foundation for subsequent analysis. Then we utilize existing DD methods to generate synthetic datasets on CMNIST-DD and CCIFAR10-DD, and evaluate their performance following the standard process. Experiments demonstrate that biases present in the original dataset significantly impact the performance of the synthetic dataset in most cases, which highlights the necessity of identifying and mitigating biases in the original datasets during DD. Finally, we reformulate DD within the context of a biased dataset.

## Datasets
We create two biased datasets for DD, named CMNIST-DD and CCFAR10-DD, following the instructions of [Nam et al.](https://proceedings.neurips.cc/paper/2020/hash/eddc3427c5d77843c2253f1e799fe933-Abstract.html). Each dataset consists of 6 training sets with varying biased ratios (0%, 10%, 50%, 80%, 95% and 100%) and 1 unbiased testing set. Finally, a parameter, severity, is introduced to regulate the intensity of disturbance on CMNIST-DD and CCFAR10-DD. 
![image](https://github.com/yaolu-zjut/Biased-DD/blob/main/biased%20dataset.JPG)

For your ease of reproducibility, we provide our datasets in the [Baidu Skydisk](https://pan.baidu.com/s/1-6uCCTc1G5icxBpVOIKeUw?pwd=bzzt). Or you can create your biased dataset using the following codes:
### CMNIST-DD
```
python biased_cifar10.py
```

### CCFAR10-DD
```
python biased_mnist_v1.py
```
### Usage
```
data = torch.load('datapath')
dst_train_img, dst_train_lab, dst_train_attr = save_data["training_imgs"], save_data["training_labs"], save_data["training_attrs"]
dst_test_img, dst_test_lab, dst_test_attr = save_data["testing_imgs"], save_data["testing_labs"], save_data["testing_attrs"]
```
## Run
We use the default hyperparameter settings of DC, DSA and DM to conduct experiments on CMNIST-DD and CCIFAR10-DD. Their official code is [here](https://github.com/VICO-UoE/DatasetCondensation).

## Acknowledgement
Our implementation references the code below, thanks to them.
* LfF: [LfF](https://github.com/alinlab/LfF)
* DatasetCondensation: [DatasetCondensation](https://github.com/VICO-UoE/DatasetCondensation)
