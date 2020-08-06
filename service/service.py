import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File

from models.experimental import attempt_load
from options import get_options
from utils.datasets import letterbox
from utils.general import (
    non_max_suppression, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device

app = FastAPI()


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Run inference
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    contents = await file.read()
    arr = np.fromstring(contents, np.uint8)
    img0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    assert img0 is not None, 'Image Failed to Load ' + file.filename
    img = letterbox(img0, new_shape=opt.img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    result = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                result.append({names[int(cls)]: dict(zip(['x', 'y', 'w', 'h'], [*xywh]))})

    return result


if __name__ == "__main__":
    opt = get_options()
    print(opt)

    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    uvicorn.run(app, host="localhost", port=5001, log_level="info")
