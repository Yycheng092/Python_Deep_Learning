#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

import torch                                        # 引入 PyTorch 的 torch (第三方)套件
from torchvision import datasets, transforms        # 從 torchvision 套件中匯入 datasets 與 transforms 兩個模組  # print(type(datasets)) print(type(transforms)) 
from torch.utils.data import DataLoader             # 從 torch.utils.data 套件中匯入 DataLoader  兩個模組  # print(type(DataLoader))

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),                        # 隨機裁剪圖片成 224x224 的大小，並且裁剪的區域大小在原圖的 70% 到 100% 之間
    transforms.RandomHorizontalFlip(p=0.5),                                     # 以 50% 的機率水平翻轉圖片
    transforms.RandomRotation(degrees=15),                                      # 隨機旋轉圖片，旋轉的角度在 -15 度到 15 度之間
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),       # 隨機調整圖片的亮度、對比度和飽和度，調整的幅度在 20% 以內
    transforms.ToTensor(),                                                      # 將圖片轉換成 PyTorch 的張量格式，並且將像素值從 0-255 的範圍縮放到 0-1 的範圍
])

# dir(transforms)  用來查看 transforms 模組中有哪些函式或類別
# dir(transforms.Compose)
# type(transforms.ToTensor)

train_dataset = datasets.CIFAR10(
    root="./data",          # 這邊是下載到硬碟當中，尚未載入到記憶體中
    train=True,
    download=True,
    transform=train_tf
)

# type(datasets.CIFAR10)
# dir(datasets.CIFAR10)
#help(datasets.CIFAR10)

'''
import os

print(os.getcwd())
# 可以觀察到 Python 提供的 datasets.CIFAR10 類別並非照片，而是存放 number 的資料結構，內容包含了一維向量的像素值以及對應的 label
'''

# 建立一個資料批次產生器，將資料集分成多個批次，每個批次包含 32 筆資料，並且在每個 epoch 結束後打亂資料順序
# 它會把 train_dataset 中的資料分成多個批次，每個批次包含 32 筆資料，因為對於 tranin_dataset 來說一次只能拿一筆資料
# 但神經網路訓練需要 32 張一起丟進模型，這樣 GPU 才能平行運算
# bratch_size 可以調整成 16、64、128 等等，根據你的 GPU 記憶體大小來決定，小的 batch 更新平繁、noise 大且收斂不穩定
# 過大的 batch 更新平穩，但泛化能力會下降
# 這只是建立一個未來要怎麼讀取資料的規則物件，並不會真正把資料讀取到記憶體中
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True
)

print("原始資料數量:", len(train_dataset))


'''
import matplotlib.pyplot as plt

img, label = train_dataset[3]   # 可以自行更改數字 0~49999 來查看不同的圖片和對應的 label

plt.imshow(img.permute(1,2,0))
plt.show()
'''

imgs, labels = next(iter(train_loader))

import torch

img1, _ = train_dataset[0]   # 第一次取第0張（會套用隨機擴增）
img2, _ = train_dataset[0]   # 第二次再取第0張（又重新隨機擴增）

print("兩次取同一張是否完全相同:", torch.equal(img1, img2))

# 資料擴增不是增加數量，而是增加資料的多樣性，讓模型在訓練過程中看到更多不同的圖片變化。