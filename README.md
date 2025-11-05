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

#Cell 6: metrics + TrainingArguments
สร้าง iou_score และ compute_metrics (ที่ Trainer จะเรียก) แล้วเตรียม TrainingArguments (eval_strategy, save_strategy, logging, fp16, load_best_model_at_end, metric_for_best_model="eval_mean_iou" )

#set arguments
```bash
args = TrainingArguments(
    output_dir=cfg.output_dir,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    learning_rate=cfg.lr,
    num_train_epochs=cfg.epochs,
    weight_decay=cfg.weight_decay,

    # สำคัญ: เวอร์ชันของคุณใช้ 'eval_strategy'
    eval_strategy=IntervalStrategy.EPOCH,  # หรือ eval_strategy="epoch"

    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=2,
    report_to=[],
    push_to_hub=False,
    remove_unused_columns=False,
    dataloader_num_workers=2,
    fp16=torch.cuda.is_available(),
    seed=cfg.seed,

    load_best_model_at_end=True,
    metric_for_best_model="eval_mean_iou",
    greater_is_better=True,
)
```

#create function to get iou score
```bash
def iou_score(pred, target, num_classes, ignore_index=255):
    mask = np.ones_like(target, dtype=bool)
    if ignore_index is not None:
        mask &= (target != ignore_index)
    ious = []
    for c in range(num_classes):
        p = (pred == c) & mask
        t = (target == c) & mask
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        ious.append(inter/union if union else 0.0)
    return np.array(ious), float(np.mean(ious)) if ious else 0.0
```
#create function compute_metrics
```bash
def compute_metrics(eval_preds):
    logits, labels = (eval_preds if isinstance(eval_preds, tuple)
                      else (eval_preds.predictions, eval_preds.label_ids))
    preds = logits.argmax(1)
    bs = preds.shape[0]; miou_list = []
    for i in range(bs):
        p = preds[i].astype(np.int32)
        y = labels[i].astype(np.int32)
        # ขนาดไม่เท่า ปรับด้วย PIL แบบ NEAREST
        if y.shape != p.shape:
            y = np.array(Image.fromarray(y).resize((p.shape[1], p.shape[0]), Image.NEAREST))
        _, miou = iou_score(p, y, NUM_LABELS, ignore_index=cfg.ignore_index)
        miou_list.append(miou)
    return {"mean_iou": float(np.mean(miou_list))}
```

# นำmask มาทับกับภาพinput เพื่อให้มองภาพง่ายขึ้น
```bash
def overlay_multi(img_pil, mask_ids, alpha=0.45):
    img = img_pil.convert("RGBA")
    h, w = mask_ids.shape
    K = int(mask_ids.max()) + 1
    cmap = plt.get_cmap("tab20")
    cols = (cmap(np.linspace(0,1,max(K,2)))[:, :3] * 255).astype(np.uint8)
    ov = np.zeros((h, w, 4), dtype=np.uint8)
    for c in range(1, K):
        m = (mask_ids == c)
        if m.any():
            ov[m,:3] = cols[c]; ov[m,3] = int(255*alpha)
    return Image.alpha_composite(img, Image.fromarray(ov))
```
#นำภาพเข้าโมเดลแล้วได้ผลลัพธ์ออกมาเป็น mask
```bash
def predict_pil(img_pil):
    x = processor(images=img_pil, return_tensors="pt").to(device)
    logits = model(**x).logits
    H, W = img_pil.size[1], img_pil.size[0]
    if logits.shape[-2:] != (H,W):
        logits = F.interpolate(logits, size=(H,W), mode="bilinear", align_corners=False)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return pred
```
