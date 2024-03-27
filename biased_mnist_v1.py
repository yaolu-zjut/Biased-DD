import copy
import os
import pickle
import random
import numpy
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import torchvision.transforms as T
from corrupted_cifar10_protocol import CORRUPTED_CIFAR10_PROTOCOL
import matplotlib.pyplot as plt
from functools import partial

mean_color = torch.load('/public/ly/my_DD/colors.th')

def colorize(raw_image, severity, attribute_label):
    std_color = [0.05, 0.02, 0.01, 0.005, 0.002][severity - 1]
    image = (
                torch.clamp(mean_color[attribute_label]
                            + torch.randn((3, 1, 1)) * std_color, 0.0, 1.0)
            ) * raw_image.unsqueeze(0).float()

    return image


COLORED_MNIST_PROTOCOL = dict()
for i in range(10):
    COLORED_MNIST_PROTOCOL[i] = partial(colorize, attribute_label=i)


class biased_mnist:
    def __init__(self, classes=10, seed=42):
        self.classes = classes
        self.set_seed(seed)

    def set_seed(self, seed=0):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        numpy.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def generate_data(self, dst, severity, biased_rate):
        protocol = COLORED_MNIST_PROTOCOL

        corrupted_imgs = []
        labs = []
        attrs = []
        corruption_labels = self.make_attr_labels(torch.LongTensor(dst.targets), bias_aligned_ratio=biased_rate, num_classes=self.classes)
        for img, target_label, corruption_label in tqdm(
                zip(dst.data, dst.targets, corruption_labels),
                total=len(corruption_labels),
        ):
            colored_img = protocol[corruption_label.item()](img, severity)
            corrupted_img = torch.from_numpy(np.array(colored_img).astype(np.uint8)).float() / 255
            corrupted_imgs.append(torch.unsqueeze(corrupted_img, dim=0))
            labs.append(target_label)
            attrs.append(corruption_label.item())

        corrupted_imgs = torch.cat(corrupted_imgs, dim=0)
        labels_all = torch.tensor(labs, dtype=torch.long)
        corruption_labels = torch.tensor(attrs, dtype=torch.long)
        return corrupted_imgs, labels_all, corruption_labels

    def make_attr_labels(self, target_labels, bias_aligned_ratio, num_classes):  # check
        num_samples_per_class = np.array(
            [
                torch.sum(target_labels == label).item()
                for label in range(num_classes)
            ]
        )
        ratios_per_class = bias_aligned_ratio * np.eye(num_classes) + (
                1 - bias_aligned_ratio
        ) / (num_classes - 1) * (1 - np.eye(num_classes))

        corruption_milestones_per_class = (
                num_samples_per_class[:, np.newaxis]
                * np.cumsum(ratios_per_class, axis=1)
        ).round()

        num_corruptions_per_class = np.concatenate(
            [
                corruption_milestones_per_class[:, 0, np.newaxis],
                np.diff(corruption_milestones_per_class, axis=1),
            ],
            axis=1,
        )

        attr_labels = torch.zeros_like(target_labels)
        for label in range(num_classes):
            indices = (target_labels == label).nonzero().squeeze()
            corruption_milestones = corruption_milestones_per_class[label]
            for corruption_idx, idx in enumerate(indices):
                attr_labels[idx] = np.min(
                    np.nonzero(corruption_milestones > corruption_idx)[0]
                ).item()

        return attr_labels


if __name__ == '__main__':
    mean = [0.1307]
    std = [0.3081]
    color_std = 0.1
    num_classes = 10
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])  # no zca
    dst_train = datasets.MNIST('/public/MountData/dataset/MNIST/', train=True, download=True, transform=transform)  # no augmentation
    dst_test = datasets.MNIST('/public/MountData/dataset/MNIST/', train=False, download=True, transform=transform)
    data = biased_mnist(classes=10)

    for aligned_rate in [0.0, 0.1, 0.5, 0.8, 0.95, 1.0]:
        for severity in [1, 2, 3, 4]:
            corrupted_testing_imgs, testing_labels, testing_corruption_labels = data.generate_data(dst_test, severity=severity, biased_rate=0)
            corrupted_training_imgs, training_labels, training_corruption_labels = data.generate_data(dst_train, severity=severity, biased_rate=aligned_rate)


            # if aligned_rate == 0.1:
            #     # for visualization
            #     bias = []
            #     t_bias = []
            #     kind = 0
            #     count = 0
            #     for i in range(len(training_labels)):
            #         if (training_labels[i] == training_corruption_labels[i]):
            #             if (training_labels[i] == kind):
            #                 t_bias.append(torch.unsqueeze(corrupted_training_imgs[i], dim=0))
            #                 count = count + 1
            #                 if (count == 5):
            #                     kind = kind + 1
            #                     count = 0
            #     t_bias = torch.cat(t_bias, dim=0)
            #     for i in range(10):
            #         bias.append(t_bias[5 * i])
            #     for i in range(10):
            #         bias.append(t_bias[5 * i + 1])
            #     for i in range(10):
            #         bias.append(t_bias[5 * i + 2])
            #     for i in range(10):
            #         bias.append(t_bias[5 * i + 3])
            #     for i in range(10):
            #         bias.append(t_bias[5 * i + 4])
            #     grid = torchvision.utils.make_grid(bias[10:40], nrow=10)
            #
            #     plt.axis('off')
            #     plt.imshow(np.transpose(grid, (1, 2, 0)))
            #     plt.tight_layout()
            #     plt.savefig('/public/MountData/dataset/Biased_dataset/CMNIST-DD/cmnist-dd_align{}_severity_{}.pdf'.format(aligned_rate, severity), bbox_inches='tight', pad_inches=0)

            save = {"testing_imgs": corrupted_testing_imgs, "testing_labs": testing_labels, "training_imgs": corrupted_training_imgs,
                    "training_labs": training_labels, "testing_attrs": testing_corruption_labels, "training_attrs": training_corruption_labels}
            torch.save(save, "/public/MountData/dataset/Biased_dataset/CMNIST-DD/MNIST_align{}_severity_{}.pth".format(aligned_rate, severity))

