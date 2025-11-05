# transformer

#structure file
```bash
project-root/
├─ data/
│ ├─ train/
│ │ ├─ images/
│ │ └─ masks/
│ └─ test/
│ ├─ images/
│ └─ masks/ # อาจไม่มี GT สำหรับ test
├─ outputs/
│ ├─ pred_masks_test/
│ ├─ pred_overlays_test/
│ └─ checkpoints/
├─ tranformer_road_rgb.py # โค้ดหลัก (หรือเป็น notebook แยกเป็นเซลล์)
└─ README.md
```

#download file from google drive and extract file
```bash
import gdown
import os

# File ID ของไฟล์ zip
file_id = "1ndCj8rEXuaQkMhfJWBclEeZAp3cbNwzR"
zip_path = "./unet_dataset.zip"

# ดาวน์โหลดไฟล์ zip
gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)

# แตกไฟล์ zip
import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./unet_dataset")

print("Download and unzip completed! Files are in ./unet_dataset")
```

## library 
```bash
!pip install "numpy<2.3.0" --force-reinstall
!pip install -U pillow matplotlib
!pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -U transformers datasets evaluate accelerate
!pip install PyDrive
!pip install -U opencv-python-headless
!pip install --upgrade gdown
```
#Cell 1: paths & pairs
กำหนดPath (ROOT / OUTPUT_DIR) แล้วสร้างฟังก์ชันสำหรับหาไฟล์ภาพและแมสก์ (list_images, normalize_stem, build_pairs) → สร้าง train_pairs และ test_pairs เป็น list ของ (image_path, mask_path)
ผลลัพธ์: ตัวแปร train_pairs / test_pairs ที่โค้ดอื่นจะใช้ต่อ
สำคัญที่ต้องแก้ก่อนรัน: ROOT ต้องชี้ไปยัง dataset ของคุณจริง ๆ

```bash
ROOT = "/workspace/unet_dataset/crack_segmentation_dataset"
OUTPUT_DIR = os.path.join(ROOT, "output_segformer")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_IMG_DIR  = os.path.join(ROOT, "train", "images")
TRAIN_MASK_DIR = os.path.join(ROOT, "train", "masks")
TEST_IMG_DIR   = os.path.join(ROOT, "test",  "images")
TEST_MASK_DIR  = os.path.join(ROOT, "test",  "masks")
```



 Cell 3: config
ประกาศ dataclass CFG เก็บ hyperparams (model_name, img_size, batch_size, epochs, lr, weight_decay, seed, ignore_index) และสร้าง cfg instance
 ผลลัพธ์: ตัวแปร cfg ที่ทุกเซลล์ใช้ต่อสามารถกำหนดค่าต่างๆได้ที่เซลล์นี้
 
```bash
 CFG = {
  "model_name": "nvidia/segformer-b1-finetuned-ade-512-512",
  "img_size": 512,
  "batch_size": 10,
  "epochs": 25,
  "lr": 3e-4,
  "weight_decay": 0.0,
  "seed": 42,
  "output_dir": "outputs/"
}
```

#Cell 5: datasets
สร้าง DatasetDict จาก train_pairs/test_pairs 

#create dataset
```bash
from datasets import Dataset, DatasetDict

def to_dataset(pairs):
    return Dataset.from_dict({"image_path":[ip for ip,_ in pairs],
                              "mask_path":[mp for _,mp in pairs]})

ds = DatasetDict({"train": to_dataset(train_pairs),
                  "test":  to_dataset(test_pairs)})
```
