import json
from pathlib import Path

DATASET_ROOT = Path("/Users/mehmetdegirmenci/Desktop/Microsoft proje/dataset")
OUTPUT_ROOT  = Path("/Users/mehmetdegirmenci/Desktop/Microsoft proje/yolo_dataset")
IMG_W, IMG_H = 640, 512
SPLITS = ["train", "val", "test"]

def convert(dataset_root, output_root):
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[SKIP] {split} bulunamadı")
            continue

        out_img = output_root / split / "images"
        out_lbl = output_root / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        total_frames = 0
        skipped = 0

        for seq in sequences:
            json_path = seq / "IR_label.json"
            if not json_path.exists():
                continue

            with open(json_path) as f:
                data = json.load(f)

            exist   = data["exist"]
            gt_rect = data["gt_rect"]
            frames  = sorted(seq.glob("*.jpg"))

            for i, frame_path in enumerate(frames):
                if i >= len(exist):
                    break

                if exist[i] != 1:
                    skipped += 1
                    continue

                bbox = gt_rect[i]
                if bbox is None:
                    skipped += 1
                    continue

                x, y, w, h = bbox

                # YOLO normalize
                x_c = (x + w / 2) / IMG_W
                y_c = (y + h / 2) / IMG_H
                w_n = w / IMG_W
                h_n = h / IMG_H

                # Kırpma: değerler 0-1 arasında olmalı
                x_c = max(0.0, min(1.0, x_c))
                y_c = max(0.0, min(1.0, y_c))
                w_n = max(0.0, min(1.0, w_n))
                h_n = max(0.0, min(1.0, h_n))

                unique_name = f"{seq.name}_{frame_path.stem}"

                # Sembolik link (kopya değil, disk tasarrufu)
                link_path = out_img / (unique_name + ".jpg")
                if not link_path.exists():
                    link_path.symlink_to(frame_path.resolve())

                # Label dosyası (class 0 = uav)
                lbl_path = out_lbl / (unique_name + ".txt")
                with open(lbl_path, "w") as f:
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

                total_frames += 1

        print(f"[{split}] {total_frames} frame etiketlendi, {skipped} frame atlandı (exist=0)")

    # YOLO config YAML
    yaml_path = output_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""path: {output_root}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['uav']
""")
    print(f"\nYAML oluşturuldu: {yaml_path}")


if __name__ == "__main__":
    convert(DATASET_ROOT, OUTPUT_ROOT)
    print("\nDönüşüm tamamlandı.")