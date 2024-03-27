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


class biased_cifar10:
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

    def generate_data(self, dst, corruption_names, severity, biased_rate):
        protocol = CORRUPTED_CIFAR10_PROTOCOL
        convert_img = T.Compose([T.ToTensor(), T.ToPILImage()])

        corrupted_imgs = []
        labs = []
        attrs = []
        corruption_labels = self.make_attr_labels(torch.LongTensor(dst.targets), bias_aligned_ratio=biased_rate, num_classes=self.classes)
        for img, target_label, corruption_label in tqdm(
                zip(dst.data, dst.targets, corruption_labels),
                total=len(corruption_labels),
        ):
            method_name = corruption_names[corruption_label]
            corrupted_img = protocol[method_name](convert_img(img), severity + 1)
            corrupted_img = torch.from_numpy(np.array(corrupted_img).astype(np.uint8)).permute(2, 0, 1).float() / 255
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
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])  # no zca
    dst_train = datasets.CIFAR10('/public/MountData/dataset/cifar10/', train=True, download=True, transform=transform)  # no augmentation
    dst_test = datasets.CIFAR10('/public/MountData/dataset/cifar10/', train=False, download=True, transform=transform)
    data = biased_cifar10(classes=10)
    for aligned_rate in [0.0, 0.1, 0.5, 0.8, 0.95, 1.0]:
        for severity in [1, 2, 3, 4]:
            corrupted_testing_imgs, testing_labels, testing_corruption_labels = data.generate_data(dst_test, corruption_names=[
                                "Snow",
                                "Frost",
                                "Fog",
                                "Brightness",
                                "Contrast",
                                "Spatter",
                                "Elastic",
                                "JPEG",
                                "Pixelate",
                                "Saturate",
                            ], severity=severity, biased_rate=0)

            corrupted_training_imgs, training_labels, training_corruption_labels = data.generate_data(dst_train, corruption_names=[
                "Snow",
                "Frost",
                "Fog",
                "Brightness",
                "Contrast",
                "Spatter",
                "Elastic",
                "JPEG",
                "Pixelate",
                "Saturate",
            ], severity=severity, biased_rate=aligned_rate)

            # if aligned_rate == 0.95:
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
            #
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
            #     plt.savefig('/public/MountData/dataset/Biased_dataset/CCIFAR10-DD/ccifar10-dd_align{}_severity_{}.pdf'.format(aligned_rate, severity), bbox_inches='tight', pad_inches=0)

            save = {"testing_imgs": corrupted_testing_imgs, "testing_labs": testing_labels, "training_imgs": corrupted_training_imgs,
                    "training_labs": training_labels, "testing_attrs": testing_corruption_labels, "training_attrs": training_corruption_labels}
            torch.save(save, "/public/MountData/dataset/Biased_dataset/CCIFAR10-DD/CIFAR10_align{}_severity_{}.pth".format(aligned_rate, severity))

