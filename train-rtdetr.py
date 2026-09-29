"""Train SFDS-DETR with Ultralytics RT-DETR."""

from ultralytics import RTDETR


if __name__ == "__main__":
    model = RTDETR("ultralytics/cfg/models/rt-detr/sfds-detr.yaml")
    model.train(
        data="ultralytics/cfg/datasets/VisDrone.yaml",
        imgsz=640,
        epochs=100,
        batch=16,
        workers=0,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=1.0,
        momentum=0.9,
        weight_decay=0.0001,
        close_mosaic=10,
        project="runs/train",
        name="SFDS-DETR",
        cache=False,
        warmup_epochs=5,
        label_smoothing=0.0,
        mosaic=0.0,
        mixup=0.0,
        flipud=0.0,
        box=7.5,
        amp=False,
    )
