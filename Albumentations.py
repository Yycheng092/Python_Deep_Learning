### 影像的資料擴增 albumentations 方法
# pip install albumentations opencv-python matplotlib

import cv2                          # 引入 OpenCV 的 cv2 物件
import matplotlib.pyplot as plt     # 引入 Matplotlib 的 pyplot 模組，並且命名為 plt
import albumentations               # 引入 Albumentations 的 albumentations 模組，並命名為 A

# 在同一個檔案路徑中插入一張名為 dog.jpg 的圖片，並使用 OpenCV 讀取這張圖片
# 當圖片成功讀取後，函式會回傳一個 NumPy 陣列
img = cv2.imread("dog.jpg")
#print(image)

# OpenCV 讀取是 BGR，需要轉 RGB
# cvtColor (convert color)   是 cv2 模組中的一個函式
# COLOR_BGR2RGB 這是 opencv 定義的一個常數，表示從 BGR 顏色空間轉換到 RGB 顏色空間，還包含其他種 1.BGR → RGB 2.BGR → Gray 3.BGR → HSV 4.RGB → Gray
# COLOR_BGR2RGB 不是方法，而是一個常數，代表一個特定的顏色轉換規則，告訴 cvtColor 函式要如何轉換圖片的顏色空間
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 建立 augmentation pipeline
transform = albumentations.Compose([
    albumentations.RandomResizedCrop(size=(224, 224)),                  # 隨機裁剪圖片成 224x224 的大小
    albumentations.HorizontalFlip(p=0.5),                               # 以 50% 的機率水平翻轉圖片
    albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),     # 隨機平移、縮放和旋轉圖片，平移的範圍在原圖的 5% 以內，縮放的範圍在 10% 以內，旋轉的角度在 -15 度到 15 度之間，並且以 70% 的機率套用這些變換
    albumentations.RandomBrightnessContrast(p=0.5),                     # 隨機調整圖片的亮度和對比度，並且以 50% 的機率套用這些變換
    albumentations.GaussNoise(p=0.3),                                   # 隨機在圖片中加入高斯噪聲，並且以 30% 的機率套用這個變換   
])

# 使用 augmentation 中的 Compse 時，會自動呼叫到 Compose.__call__() 這個函式，其中.__call__() 函是參數包含了 image 這個參數 
# def __call__(self, image=None, mask=None, bboxes=None, keypoints=None):
# 其中我只有用到 image 這個參數，所以在呼叫 transform(image=img) 的時候，會把 img 這張圖片傳入到 Compose.__call__() 這個函式中，
# 若有使用到  mask 則會這樣寫，transform(image=image,mask=mask)
augmented = transform(image=img)

# Albumentations 可能同時回傳多種資料，所以用 dictionary 包起來
# 而我只需要取用 image 這個 key 對應的 value 就好，所以寫 augmented["image"] 就可以取出來了
aug_image = augmented["image"]

# 顯示圖片
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Augmented")
plt.imshow(aug_image)
plt.axis("off")

plt.show()