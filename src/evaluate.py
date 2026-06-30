import motmetrics as mm
import numpy as np 
import time

def evaluate(tracker, model, seqeunces, conf=0.3):
    accuracy = mm.MOTAccumulator(auto_id=True)
    fps_list = []

    for seq_path , json_path in seqeunces:
        import json 
        from pathlib import Path

        with open(json_path) as f:
            data = json.load(f)
        
        exist = data["exist"]
        gt_rect = data["gt_rect"]
        frames = sorted(Path(seq_path).glob("*.jpg"))

        tracker_seq = type(tracker)()

        for i, frame_path in enumerate(frames):
            if i >= len(exist):
                break

            gt_ids, gt_boxes = [], []

            if exist[i] == 1 and gt_rect[i] is not None:
                x, y, w, h = gt_rect[i]
                cx = x+ w/2
                cy = y + h/2
                gt_ids.append(0)
                gt_boxes.append([cx, cy])

            t0 = time.time()
            results = model(str(frame_path), conf=conf, verbose= False)
            boxes = results[0].boxes

            detections = []
            for box in boxes.xyxy :
                x1,y1,x2,y2 = box.tolist()
                detections.append(((x1+x2)/2, (y1+y2)/2))

            tracks = tracker_seq.update(detections)
            fps_list.append(1.0 / (time.time()-t0))

            pred_ids = [t.id for t in tracks]
            pred_boxes = [[t.kalman.state[0], t.kalman.state[1]] for t in tracks]

            distances = mm.distances.norm2squared_matrix(gt_boxes,pred_boxes, max_d2=10000)

            accuracy.update(gt_ids,pred_ids,distances)
    
    mh = mm.metrics.create()
    summary = mh.compute(accuracy, metrics=["mota","idf1","num_switches"], name="eval")

    print(summary.to_string())
    print(f"FPS: {np.mean(fps_list):.1f}")
    return summary




