# -*-coding:utf-8-*-
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(self, name_list, img):
    showimg = img
    with torch.no_grad():
        img = letterbox(img, new_shape=self.opt.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = self.model(img, augment=self.opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)
        info_show = ""
        # Process detections
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    name_list.append(self.names[int(cls)])
                    # single_info = Annotator.box_label(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
                    # single_info = Annotator.box_label(xyxy, showimg, label=label, color=self.colors[int(cls)])

                    single_info = plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)],
                                               line_thickness=2)

                    # print(single_info)
                    # info_show = info_show + single_info + "\n"
    return info_show

if __name__ == '__main__':
    img = '/data/images/zidane.jpg'
    name_list = []
    result_im = detect(name_list, img)
    print(result_im)