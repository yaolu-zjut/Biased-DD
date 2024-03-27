# Biased-DD
A repository of [Exploring the Impact of Dataset Bias on Dataset Distillation](https://arxiv.org/pdf/2403.16028.pdf)

## Abstract
Dataset Distillation (DD) is a promising technique to synthesize a smaller dataset that preserves essential information from the original dataset. This synthetic dataset can serve as a substitute for the original large-scale one, and help alleviate the training workload. However, current DD methods typically operate under the assumption that the dataset is unbiased, overlooking potential bias issues within the dataset itself. To fill in this blank, we systematically investigate the influence of dataset bias on DD. To the best of our knowledge, this is the first exploration in the DD domain. Given that there are no suitable biased datasets for DD, we first construct two biased datasets, CMNIST-DD and CCIFAR10-DD, to establish a foundation for subsequent analysis. Then we utilize existing DD methods to generate synthetic datasets on CMNIST-DD and CCIFAR10-DD, and evaluate their performance following the standard process. Experiments demonstrate that biases present in the original dataset significantly impact the performance of the synthetic dataset in most cases, which highlights the necessity of identifying and mitigating biases in the original datasets during DD. Finally, we reformulate DD within the context of a biased dataset.

## Datasets
We create two biased datasets for DD, named CMNIST-DD and CCFAR10-DD, following the instructions of [Nam et al.](). Each dataset consists of 6 training sets with varying biased ratios (0%, 10%, 50%, 80%, 95% and 100%) and 1 unbiased testing set. Finally, a parameter, severity, is introduced to regulate the intensity of disturbance on CMNIST-DD and CCFAR10-DD.

## 
```
function test() {
  console.log("notice the blank line before this function?");
}
```

## Acknowledgement
Our implementation references the code below, thanks to them.
