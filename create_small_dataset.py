import json
from pathlib import Path

DATASET_ROOT = Path("/Users/mehmetdegirmenci/Desktop/Microsoft proje/dataset")
OUTPUT_ROOT  = Path("/Users/mehmetdegirmenci/Desktop/Microsoft proje/yolo_dataset_small")
IMG_W, IMG_H = 640, 512

# Kaç sequence kullanılacak
N_TRAIN = 20
N_VAL   = 10
N_TEST  = 10

SPLITS = {
    "train": N_TRAIN,
    "val":   N_VAL,
    "test":  N_TEST,
}

for split, n_seq in SPLITS.items():
    split_dir = DATASET_ROOT / split
    out_img = OUTPUT_ROOT / split / "images"
    out_lbl = OUTPUT_ROOT / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])[:n_seq]
    total = 0

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
            if i >= len(exist) or exist[i] != 1:
                continue
            bbox = gt_rect[i]
            if bbox is None:
                continue

            x, y, w, h = bbox
            x_c = max(0.0, min(1.0, (x + w / 2) / IMG_W))
            y_c = max(0.0, min(1.0, (y + h / 2) / IMG_H))
            w_n = max(0.0, min(1.0, w / IMG_W))
            h_n = max(0.0, min(1.0, h / IMG_H))

            unique_name = f"{seq.name}_{frame_path.stem}"

            link_path = out_img / (unique_name + ".jpg")
            if not link_path.exists():
                link_path.symlink_to(frame_path.resolve())

            lbl_path = out_lbl / (unique_name + ".txt")
            with open(lbl_path, "w") as f:
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

            total += 1

    print(f"[{split}] {len(sequences)} sequence, {total} frame")

yaml_path = OUTPUT_ROOT / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"""path: {OUTPUT_ROOT}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['uav']
""")
print("\nHazır:", OUTPUT_ROOT)
