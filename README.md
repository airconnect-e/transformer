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

#Cell 1: paths & pairs
กำหนดPath (ROOT / OUTPUT_DIR) แล้วสร้างฟังก์ชันสำหรับหาไฟล์ภาพและแมสก์ (list_images, normalize_stem, build_pairs) → สร้าง train_pairs และ test_pairs เป็น list ของ (image_path, mask_path)
ผลลัพธ์: ตัวแปร train_pairs / test_pairs ที่โค้ดอื่นจะใช้ต่อ
สำคัญที่ต้องแก้ก่อนรัน: ROOT ต้องชี้ไปยัง dataset ของคุณจริง ๆ

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

## Cell 2:
กำหนด จำนวนclass labels เป็น 2
และค่าสีเป็น 255
สร้างฟังกันชั่น remap_mask_to_contiguous เพื่ออ่านค่าสีและคืนค่ากลับมาเป็นตัวเลข

 cell 2.5:
 อ่านเลขที่ได้จากการ map เห็นค่าสีที่อ่านได้เป็น 0 1

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


#Cell 4: processor & model
 โหลด AutoImageProcessor และ SegformerForSemanticSegmentation จาก Hugging Face แล้วนำโมเดลไปรันด้วย GPU (ถ้ามี)
 ผลลัพธ์: processor และ model พร้อมใช้
 
#Cell 5: datasets
สร้าง DatasetDict จาก train_pairs/test_pairs แล้วใช้ _load_example map ทุกตัวเพื่อ:
เปิดภาพ → convert RGB
ถ้ามี mask → โหลด, เปลี่ยนขนาดเป็น cfg.img_size และ remap labels → ใช้ processor(images=..., segmentation_maps=[m]) เพื่อได้ pixel_values และ labels เป็น tensor
ตั้ง ds.set_format(type="torch", columns=["pixel_values","labels"])
 ผลลัพธ์: ds["train"], ds["test"] พร้อมให้ Trainer ใช้งาน

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

#create function iou_score
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

#Cell 7: training (Trainer + Callback)
กำหนด LogTrainMetricsEachEpoch callback เพื่อ evaluate บน train_dataset หลังจบ epoch แล้ว log ผล (เพื่อดู train mIoU ทันที)
สร้าง Trainer (model, args, train_dataset, eval_dataset=ds["test"], processing_class=processor, compute_metrics=compute_metrics)
เรียก trainer.train() และบันทึก model + processor ไปที่ cfg.output_dir

#Cell 8: inference & save
ฟังก์ชัน overlay_multi สร้างภาพ overlay แบบ multi-class (ใช้ colormap และใส่ alpha)
predict_pil(img_pil) preprocess ผ่าน processor, ทำ forward model → ปรับ logits ให้ match ขนาดรูปจริง ถ้าต่างด้วย F.interpolate → คืน pred mask (argmax)
ลูปผ่าน test_pairs แล้วเก็บ pred mask และ overlay image ลงโฟลเดอร์ (pred_masks_test, pred_overlays_test)

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

#Cell 9: per-image metrics & summary
ฟังก์ชัน confusion_from_pair แปลง pred/gt เป็น confusion matrix, metrics_from_cm คำนวณ mIoU ต่อคลาสและ summary
วนไฟล์ใน test_pairs, โหลด pred จากไฟล์ที่เซฟไว้ หรือ predict ใหม่ → ถ้ามี GT จะ remap และคำนวณ CM แล้วเขียน per-image csv และรวม CM เป็น cm_total → สร้าง summary CSV (metrics_test_summary.csv)

Cell 10: debug sample overlay (train & test)
แสดงตัวอย่างภาพ input / GT / GT overlay / Pred overlay สำหรับ SAMPLES_TO_SHOW ตัวอย่างจาก train_pairs (10)
พร้อม print class distribution ใน pred (uniq, counts)
 ประโยชน์: ตรวจสอบคุณภาพผลพยากรณ์แบบเร็ว ๆ และจับปัญหาเช่น class collapse (pred เป็นคลาสเดียว) หรือ mask ถูกยืด/บิด

Cell 11
แสดงตัวอย่างภาพ input / GT / GT overlay / Pred overlay สำหรับ SAMPLES_TO_SHOW ตัวอย่างจาก test_pairs (11)
พร้อม print class distribution ใน pred (uniq, counts)
 ประโยชน์: ตรวจสอบคุณภาพผลพยากรณ์แบบเร็ว ๆ และจับปัญหาเช่น class collapse (pred เป็นคลาสเดียว) หรือ mask ถูกยืด/บิด

Cell 12
นิยาม metrics_from_cm(cm) ที่คืน per-class precision/recall/f1/iou/acc และ summary (mPrecision, mRecall, mF1, mIoU, mAcc, overall_acc).
metrics_table(cm, class_names, precision_digits) สร้าง DataFrame ที่แสดงค่าด้วย precision สูงสุด (ตัวเลขยาว ๆ) — เพื่อ debug เลขทศนิยมแบบเต็ม.

Cell 13
พิมพ์ per_class precision และ summary จาก metrics_from_cm(cm) — ใช้ตรวจค่าที่ได้แบบ raw.

Cell 14
อ่าน trainer.state.log_history แล้วแยก train_loss และ eval_loss ที่ถูก log เพื่อพล็อตกราฟเปรียบเทียบ (train vs eval) — ช่วยดูว่า overfit หรือไม่

Cell 15
วนเทสบน train_pairs, ทำ predict, ถ้ามี GT คำนวณ confusion, เก็บ metrics_train_per_image.csv และ metrics_train_summary.csv.
มีฟังก์ชัน train_risk_hint(summary_path, high_bar=0.85) ให้ hint ว่า train mIoU สูงเกินไปอาจเป็นสัญญาณ overfit.

Cell 16
วนทดสอบ test_pairs, สร้าง metrics_test_per_image.csv และ metrics_test_summary.csv.
test_hint(summary_path, ok_bar=0.70) 

Cell 17
โหลด summary ของ train/test แล้วเรียก diagnose_overfit(train_sum, test_sum, high_bar=0.85, gap_bar=0.10) ซึ่งจะคืนสถานะ overfit / underfit? / ok ขึ้นกับค่า mIoU และ gap ระหว่าง train/test.

Cell 18
โหลด class_mapping.json ถ้ามี เพื่อรองรับ mapping ที่ซับซ้อน (value2index/index2value).
นิยามใหม่ของ remap_mask_to_contiguous, predict_pil, overlay_multi (พร้อม edge detection), และ metrics helper — เขียนผลลัพธ์ pred mask, overlay, CSV per-image, และ summary test metrics.
ถ้ามี GT จะคำนวณ mIoU, precision, recall, f1, overall_acc และบันทึก. หากไม่มี GT จะเขียน note ว่า "No GT available"

Cell 19 
แปลง log_history เป็น DataFrame, เติมคอลัมน์ที่อาจขาด แล้วดึงค่าล่าสุดของแต่ละ epoch (last log per epoch) เป็น per_epoch.
แสดง per_epoch เป็นตารางก่อนพล็อต.

Cell 20
เรียก trainer.evaluate(eval_dataset=ds["test"], metric_key_prefix="test") แล้วพิมพ์ loss/iou ที่ได้ — เป็น checkpoint check สุดท้าย

Cell 21
พล็อต per_epoch["loss"] กับ per_epoch["eval_loss"] ถ้ามี พร้อม grid/legend เป็นกราฟดู overfit

Cell 22
รวมการ plot: loss, eval loss, horizontal line ของ test loss จาก test_metrics ถ้ามี, และ plot mIoU (train/eval) กรณีที่คอลัมน์ชื่ออาจต่างกัน — พยายามหาชื่อคอลัมน์ที่มี "iou" เพื่อพล็อต.
สรุป: มีสองกราฟหลักให้ดูทั้ง loss และ mIoU ต่อ epoch เพื่อประเมิน training dynamics.
