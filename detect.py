# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_img_size, colorstr, increment_path, scale_boxes, xyxy2xywh,
    non_max_suppression,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / "runs/train/epoch=60bs=8/weights/best.pt",  # ROOT / "yolov5s.pt"
        source=ROOT / "data/from_Airsim/photos_1715246442",  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / "data/VisDrone.yaml",  # dataset.yaml path
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        imgsz=(640, 640),  # inference size (height, width)
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        nosave=True,  # 不保存检测结果
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,
        project=ROOT / "runs/detect",  # 保存结果的项目文件夹路径
        name="result_airsim",  # save results to project/name
        exist_ok=False,  # 如果结果文件夹已存在，是否覆盖

):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    # path是图像或视频文件的路径，im是图像数据，im0s是原始图像数据，vid_cap是视频捕获对象（如果源是视频的话），s是用于记录日志的字符串。
    for path, im, im0s, vid_cap, s in dataset:
        # 第一部分准备图像
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)
        # 检测
        with dt[1]:

            if model.xml and im.shape[0] > 1:  # 如果是xml，且输入图像数量大于1
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:  # TODO：这里好像只执行了else，所以上面的可以删
                pred = model(im) # 调试得im(1*3*640*640) im0s(1280*1280*3)
        # print(len(pred), pred[0].shape) # 2 torch.Size([1, 25200, 15])
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # print(len(pred), pred[0].shape) # 1 torch.Size([8, 6])
        # 可见non_max_suppression将pred进行了筛选，删除重叠的部分。

        # 遍历每个预测结果。pred是模型的预测结果，i是当前预测结果的索引，det是当前预测结果。
        for i, det in enumerate(pred):  # pred包含了很多图片各自的det，是list
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # 路径转换为path对象

            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # txt保存路径
            s += "%gx%g " % im.shape[2:]  # 将图像的高度和宽度添加到日志字符串s中
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 将预测边界框从模型输入大小转回原始图像大小，但这个在保存txt时需要

            if len(det):  # 如果检测结果不为0
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 遍历det中所有预测结果的类别ID（不重复）
                # det正常应该是对象数*6，比如5*6 8*6。6中前4个为坐标，然后是置信度和类别
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    #print(c, n)
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 输出例如3 dogs ， 1 dog 的信息

                # Write results 遍历每个预测结果，将其写入txt文件。
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file 需要
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

        # Print time (inference-only) 记录信息
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results 记录一些关于模型性能信息和结果保存的信息，也会显示在控制台上
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


'''
def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    # 检查环境是否满足要求
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run()

'''
if __name__ == "__main__":
    run()
