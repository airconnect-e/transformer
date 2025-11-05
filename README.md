# transformer
Cell 1: paths & pairs
กำหนดPath (ROOT / OUTPUT_DIR) แล้วสร้างฟังก์ชันสำหรับหาไฟล์ภาพและแมสก์ (list_images, normalize_stem, build_pairs) → สร้าง train_pairs และ test_pairs เป็น list ของ (image_path, mask_path)
 ผลลัพธ์: ตัวแปร train_pairs / test_pairs ที่โค้ดอื่นจะใช้ต่อ
 สำคัญที่ต้องแก้ก่อนรัน: ROOT ต้องชี้ไปยัง dataset ของคุณจริง ๆ
 จุดบกพร่องที่พบบ่อย:
การจับคู่ชื่อไฟล์พังได้ง่ายถ้านามสกุลหรือ suffix ของแมสก์ไม่ตรงที่ normalize_stem คาดไว้ → ถ้าคู่ไม่ครบจะสร้าง mp = "" (no mask)
โครงสร้างโฟลเดอร์ต้องตรงกับคอนฟิก (train/images, train/masks,)
 แก้ไข: หากชื่อไฟล์ไม่ตรง เพิ่ม pattern ใน normalize_stem หรือแก้ให้ build_pairs ใช้ matching แบบอื่น

 Cell 2:
กำหนด จำรวนclass labels เป็น 2
และค่าสีเป็น 255
สร้างฟังกันชั่น remap_mask_to_contiguous เพื่ออ่านค่าสีและคืนค่ากลับมาเป็นตัวเลข

 cell 2.5:
 อ่านเลขที่ได้จากการ map เห็นค่าสีที่อ่านได้เป็น 0 1

 Cell 3: config
ประกาศ dataclass CFG เก็บ hyperparams (model_name, img_size, batch_size, epochs, lr, weight_decay, seed, ignore_index) และสร้าง cfg instance
 ผลลัพธ์: ตัวแปร cfg ที่ทุกเซลล์ใช้ต่อสามารถกำหนดค่าต่างๆได้ที่เซลล์นี้

 Cell 4: processor & model
 โหลด AutoImageProcessor และ SegformerForSemanticSegmentation จาก Hugging Face แล้วนำโมเดลไปรันด้วย GPU (ถ้ามี)
 ผลลัพธ์: processor และ model พร้อมใช้
 จุดต้องระวัง:
ignore_mismatched_sizes=True จะโหลดโมเดลแม้ head จำนวนคลาสไม่ตรง — แต่หัวอาจสุ่ม init ใหม่ → ต้อง fine-tune จริงจัง
ตรวจสอบ device (cuda/ cpu) — ถ้าไม่มี GPU ควรลด batch size /ใช้ CPU optimized flow

Cell 5: datasets
สร้าง DatasetDict จาก train_pairs/test_pairs แล้วใช้ _load_example map ทุกตัวเพื่อ:
เปิดภาพ → convert RGB
ถ้ามี mask → โหลด, เปลี่ยนขนาดเป็น cfg.img_size และ remap labels → ใช้ processor(images=..., segmentation_maps=[m]) เพื่อได้ pixel_values และ labels เป็น tensor
ตั้ง ds.set_format(type="torch", columns=["pixel_values","labels"])
 ผลลัพธ์: ds["train"], ds["test"] พร้อมให้ Trainer ใช้งาน
 ปัญหาที่เจอบ่อย:
การ map แบบไม่เป็น batch อาจช้า ถ้า dataset ใหญ่ให้ใช้ batched=True หรือ DataLoader on-the-fly
การ resize mask เป็น cfg.img_size อาจทำให้ aspect ratio เพี้ยน — แต่จำเป็นสำหรับ model ที่คาดขนาดคงที่
memory spike ถ้า processor คืน tensor ขนาดใหญ่ทั้งหมดใน RAM — ใช้ dataloader แบบ streaming หรือ ลด img_size

Cell 6: metrics + TrainingArguments
สร้าง iou_score และ compute_metrics (ที่ Trainer จะเรียก) แล้วเตรียม TrainingArguments (eval_strategy, save_strategy, logging, fp16, load_best_model_at_end, metric_for_best_model="eval_mean_iou" )
 สิ่งสำคัญที่ต้องระวัง:
compute_metrics คืนค่า key "mean_iou" แต่ TrainingArguments.metric_for_best_model ตั้งเป็น "eval_mean_iou" — อาจทำให้ load_best_model_at_end ไม่ทำงาน เพราะชื่อ metric ไม่ตรงกัน → แก้เป็นชื่อที่ Trainer จะ log (เช่น "eval_mean_iou") หรือปรับ compute_metrics ให้คืน {"eval_mean_iou": ...}
fp16=True จะใช้ half precision ถ้า GPU รองรับ; แต่บางโมเดล/ops อาจเกิด NaN — ปิดถ้ามีปัญหา

Cell 7: training (Trainer + Callback)
กำหนด LogTrainMetricsEachEpoch callback เพื่อ evaluate บน train_dataset หลังจบ epoch แล้ว log ผล (เพื่อดู train mIoU ทันที)
สร้าง Trainer (model, args, train_dataset, eval_dataset=ds["test"], processing_class=processor, compute_metrics=compute_metrics)
เรียก trainer.train() และบันทึก model + processor ไปที่ cfg.output_dir
 จุดเสี่ยง / หมายเหตุ:
processing_class=processor เป็น parameter ที่ไม่ค่อยได้ใช้ — ปกติ Trainer รับ tokenizer หรือ data_collator — แต่ถ้าเป้าหมายคือให้ Trainer know how to preprocess อาจต้องแน่ใจว่า interaction ถูกต้อง
การ evaluate บน trainer.train_dataset ทุก epoch จะชะลอการเทรน (เพิ่ม time) — tradeoff ระหว่างข้อมูล debug กับเวลา
OOM: ถ้ารันแล้วระเบิด ให้ลด per_device_train_batch_size หรือใช้ gradient_accumulation_steps เพื่อจำลอง batch ใหญ่

Cell 8: inference & save
ฟังก์ชัน overlay_multi สร้างภาพ overlay แบบ multi-class (ใช้ colormap และใส่ alpha)
predict_pil(img_pil) preprocess ผ่าน processor, ทำ forward model → ปรับ logits ให้ match ขนาดรูปจริง ถ้าต่างด้วย F.interpolate → คืน pred mask (argmax)
ลูปผ่าน test_pairs แล้วเก็บ pred mask และ overlay image ลงโฟลเดอร์ (pred_masks_test, pred_overlays_test)
 ข้อสังเกตสำคัญ:
ขนาด H/W ต้องระวัง: PIL.Image.size ให้ (W,H) — โค้ดเอา H,W ถูกต้องด้วย img_pil.size[1], img_pil.size[0] แต่ถ้แก้ผิดอาจ swap axis ทำให้ output เพี้ยน
ถ้ logits ขนาดไม่ตรง จะใช้ bilinear resize — ทำให้ขอบ/คลาสเพี้ยนเล็กน้อย; ถาต้องการความถูกต้องสูงต้องให้ model ช่วงอินพุตเป็นขนาดคงที่ตอน training/inference

Cell 9: per-image metrics & summary
ฟังก์ชัน confusion_from_pair แปลง pred/gt เป็น confusion matrix, metrics_from_cm คำนวณ mIoU ต่อคลาสและ summary
วนไฟล์ใน test_pairs, โหลด pred จากไฟล์ที่เซฟไว้ หรือ predict ใหม่ → ถ้ามี GT จะ remap และคำนวณ CM แล้วเขียน per-image csv และรวม CM เป็น cm_total → สร้าง summary CSV (metrics_test_summary.csv)
 จุดต้องระวัง: IGNORE_INDEX ต้องถูกตั้งให้ตรงตอน remap (ไม่งั้น pixels บางส่วนหลุดออกจากการคำนวณ)
 ผลลัพธ์: 2 ไฟล์ CSV: per-image และ summary (mIoU) — ใช้ดู performance รายภาพ / รวม

Cell 10: debug sample overlay (train & test)
แสดงตัวอย่างภาพ input / GT / GT overlay / Pred overlay สำหรับ SAMPLES_TO_SHOW ตัวอย่างจาก train_pairs (10)
พร้อม print class distribution ใน pred (uniq, counts)
 ประโยชน์: ตรวจสอบคุณภาพผลพยากรณ์แบบเร็ว ๆ และจับปัญหาเช่น class collapse (pred เป็นคลาสเดียว) หรือ mask ถูกยืด/บิด
 ถ้าผลเพี้ยนดูตรงนี้ก่อน:
ค่าที่พิมพ์ classes จะบอกว่าค่าที่ pred เกิดอะไรขึ้นบ้าง (เช่น มีแต่ 0 หรือ 255) → ชี้ว่าปัญหา threshold/logit หรือ mapping

Cell 11
แสดงตัวอย่างภาพ input / GT / GT overlay / Pred overlay สำหรับ SAMPLES_TO_SHOW ตัวอย่างจาก test_pairs (11)
พร้อม print class distribution ใน pred (uniq, counts)
 ประโยชน์: ตรวจสอบคุณภาพผลพยากรณ์แบบเร็ว ๆ และจับปัญหาเช่น class collapse (pred เป็นคลาสเดียว) หรือ mask ถูกยืด/บิด
 ถ้าผลเพี้ยนดูตรงนี้ก่อน:
ค่าที่พิมพ์ classes จะบอกว่าค่าที่ pred เกิดอะไรขึ้นบ้าง (เช่น มีแต่ 0 หรือ 255) → ชี้ว่าปัญหา threshold/logit หรือ mapping

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
test_hint(summary_path, ok_bar=0.70) ให้คำแนะนำว่า mIoU >= 0.7 ถือว่า generalization พอใช้.

Cell 17
โหลด summary ของ train/test แล้วเรียก diagnose_overfit(train_sum, test_sum, high_bar=0.85, gap_bar=0.10) ซึ่งจะคืนสถานะ overfit / underfit? / ok ขึ้นกับค่า mIoU และ gap ระหว่าง train/test.
พิมพ์ตัวเลขหลักเพื่อช่วยตัดสิน.

Cell 18
โหลด class_mapping.json ถ้ามี เพื่อรองรับ mapping ที่ซับซ้อน (value2index/index2value).
นิยามใหม่ของ remap_mask_to_contiguous, predict_pil, overlay_multi (พร้อม edge detection), และ metrics helper — เขียนผลลัพธ์ pred mask, overlay, CSV per-image, และ summary test metrics.
ถ้ามี GT จะคำนวณ mIoU, precision, recall, f1, overall_acc และบันทึก. หากไม่มี GT จะเขียน note ว่า "No GT available" — ค่อนข้างครบสำหรับ production inference pipeline.

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
