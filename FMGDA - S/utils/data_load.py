from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import torch
import struct
import random
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorflow.keras.datasets import mnist

# 1. 加载 TensorFlow 的 MNIST 数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2. 定义与目标相同的预处理流程
transform = transforms.Compose([
    transforms.ToPILImage(),  # 将 numpy 数组转换为 PIL 图像
    transforms.Pad(4, fill=0, padding_mode='constant'),  # 填充为 36x36
    transforms.RandomRotation(10),  # 随机旋转±10度
    transforms.RandomAffine(0, shear=10),  # 随机仿射变换
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量 [0,1] 范围
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 标准归一化
])


# 3. 创建自定义数据集类
class CustomMNIST(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label1 = self.labels[idx]
        label2 = label1
        
        if self.transform:
            image = self.transform(image)

        return image, (label1,label2)


# 4. 创建训练集和测试集
train_dataset = CustomMNIST(train_images, train_labels, transform=transform)
test_dataset = CustomMNIST(test_images, test_labels, transform=transform)

'''
数据生成，将mnist数据拼接
'''
# 读取MNIST图像和标签的函数
def CompositeDataset(output_dir="./multi_mnist_data", num_samples=60000,train=True,save_images=False):
    """生成 MultiMNIST 数据集"""
    # 确保输出目录存在
    if not os.path.exists(output_dir) and save_images is True:
        os.makedirs(output_dir)

    if train is True:
        # 加载 MNIST 数据集
        mnist = train_dataset
    else:
        mnist = test_dataset
    # 加载 MNIST 数据集
    # mnist = datasets.MNIST(
    #     root=r'C:\Users\15834\PycharmProjects\FMGDA\FMGDA\Mnist',
    #     train=train,
    #     download=False,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),  # 转换为张量 [0,1] 范围
    #         # transforms.ToPILImage(),  # 将 numpy 数组转换为 PIL 图像
    #         transforms.Pad(4, fill=0, padding_mode='constant'),  # 填充为 36x36
    #         transforms.RandomRotation(10),  # 随机旋转±10度
    #         transforms.RandomAffine(0, shear=10),  # 随机仿射变换
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
    #         transforms.RandomHorizontalFlip(),  # 随机水平翻转
    #         transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 标准归一化
    #     ])
    # )

    # 将数据集按类别分组
    class_images = {i: [] for i in range(10)}
    for img, label in mnist:
        class_images[label].append(img)

    # 初始化图片和标签列表
    images = []
    labels = []

    # 生成 num_samples 个样本
    for i in range(num_samples):
        # 随机选择两个不同的类别
        label1 = np.random.randint(0, 10)
        label2 = label1
        # while label2 == label1:
        #     label2 = np.random.randint(0, 10)

        # 随机选择两个不同类别的图片
        img1 = class_images[label1][np.random.randint(0, len(class_images[label1]))]
        img2 = class_images[label2][np.random.randint(0, len(class_images[label2]))]

        # 将图片转换为 numpy 数组
        img1_np = img1.numpy().squeeze()
        img2_np = img2.numpy().squeeze()

        # 生成 36x36 的画布
        canvas = np.zeros((36, 36), dtype=np.float32)

        # 随机移动 img1 和 img2，最多四个像素
        def random_shift(img):
            if img is img1_np:
                shift_x = np.random.randint(-7.5, -5.5)
                shift_y = np.random.randint(-7.5, -5.5)
                # shift_x = np.random.randint(6, 8)
                # shift_y = np.random.randint(6, 8)
                shifted_img = np.roll(img, shift_x, axis=1)
                shifted_img = np.roll(shifted_img, shift_y, axis=0)
            else:
                # shift_x = np.random.randint(-8, -6)
                # shift_y = np.random.randint(-8, -6)
                shift_x = np.random.randint(5.5, 7.5)
                shift_y = np.random.randint(5.5, 7.5)
                shifted_img = np.roll(img, shift_x, axis=1)
                shifted_img = np.roll(shifted_img, shift_y, axis=0)
            return shifted_img

        img1_shifted = random_shift(img1_np)
        img2_shifted = random_shift(img2_np)

        # 叠加两张图片
        combined_img = np.minimum(img1_shifted + img2_shifted, 1.0)  # 防止像素值超过 1.0

        # 将图片转换回 Tensor
        combined_img = torch.from_numpy(combined_img).unsqueeze(0)

        # 保存图像和标签
        if save_images:
            img_path = os.path.join(output_dir, f"sample_{i}.png")
            label_path = os.path.join(output_dir, f"sample_{i}_label.txt")
            Image.fromarray((combined_img.numpy().squeeze() * 255).astype(np.uint8)).save(img_path)
            with open(label_path, 'w') as f:
                f.write(f"{label1},{label2}")

        # 将图片和标签添加到对应列表中
        images.append(combined_img)
        labels.append((label1, label2))

    # 返回图片列表和标签列表
    return images, labels

# 数据初始化
class generate_multi_mnist(Dataset):
    def __init__(self, output_dir="./multi_mnist_data", num_samples=60000,train=True,save_images=False):
        self.images, self.labels =CompositeDataset(num_samples=num_samples,output_dir=output_dir,train=train,save_images=save_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image = torch.tensor(self.images[idx].reshape(36 * 36))  # 展平为1296维
        image = torch.tensor(self.images[idx])  # 利用卷积
        label1 = torch.tensor(self.labels[idx][0], dtype=torch.long)
        label2 = torch.tensor(self.labels[idx][1], dtype=torch.long)
        return image, (label1, label2)


# ================== 数据分配函数 ==================
def split_data_to_servers(dataset, num_servers=5):
    """将数据集均匀分配到多个服务器"""
    total_size = len(dataset)
    indices = np.random.permutation(total_size)  # 随机打乱
    chunk_size = total_size // num_servers

    server_data = {}
    for i in range(num_servers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_servers - 1 else total_size
        server_data[i] = Subset(dataset, indices[start:end])

    return server_data


# 重复标签数据集
class DuplicatedLabelMNIST(Dataset):
    """生成 DuplicatedMNIST 数据集"""
    def __init__(self, train=True):
        # 加载原始 MNIST 数据集
        self.original_dataset = datasets.MNIST(root='./Mnist', train=train, download=False, transform=transforms.Compose([
            transforms.Pad(4, fill=0, padding_mode='constant'),  # 将图片调整为 36x36
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
        # 复制标签：将每个标签转换为 [label, label]
        self.labels = torch.stack([self.original_dataset.targets,
                                  self.original_dataset.targets], dim=1)
        self.images = self.original_dataset.data

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 直接通过 original_dataset 获取图像（已应用 transform）
        image, _ = self.original_dataset[idx]  # image 已转换为 Tensor
        label1 = self.labels[idx][0]  # 标签为 [2] 的 Tensor
        label2 = self.labels[idx][1]
        return image, (label1,label2)

if __name__ == "__main__":
    CompositeDataset(num_samples=5, save_images=True)
