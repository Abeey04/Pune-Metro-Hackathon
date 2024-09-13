import argparse
import os
import platform
import sys
from pathlib import Path
from numpy import random
import math
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from typing import List
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  #root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.torch_utils import TracedModel
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression_det,non_max_suppression_kpt,non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box,plot_one_box,output_to_keypoint, plot_skeleton_kpts
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode
from models.experimental import attempt_load
from shapely.geometry import Point, Polygon


class area:
    def __init__(self, contour):
            self.contour = np.array(contour, dtype=np.int32)
            self.count = 0
def point_intersection_shapely(polygon,point):
    p1 = Polygon(polygon)
    return p1.contains(Point(point))


def drawAreas(img,areas):
    for area in areas:
        color = (255,0,0)
        cv2.polylines(img, [area.contour],True,color,2)

def collinear(p0, p1, p2, point_closeness_threshold = 6, angle_closeness_threshold = .1):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p1[0], p2[1] - p1[1]
    #check if points are close enough to be considered co-linear
    if (abs(x1) < point_closeness_threshold and abs(y1) < point_closeness_threshold) or (abs(x2) < point_closeness_threshold and abs(y2) < point_closeness_threshold):
        return True
    m1 = y1/(x1+1e-12)
    m2 = y2/(x2+1e-12)
    delta_The = math.atan((m2-m1)/(1+m1*m2))
    #check if the angle between the lines is low enough to be considered co-linear
    return abs(delta_The) < angle_closeness_threshold

def simplify_by_angle(poly_in):
    coords = poly_in.exterior.coords[:]
    #loop through every point, checking if it is colinear with its neighbors, and removing if it is
    i=1
    while i < len(coords)+1:
        n = len(coords)
        if collinear(coords[(i-1)%n],coords[i%n],coords[(i+1)%n]):
            del coords[i%n]
            i-=1
        i+=1

    return coords
        


@smart_inference_mode()
def run(
        weights1=ROOT / 'yolov5s-seg.pt',  # model1.pt path(s)
        weights2=ROOT / 'yolov7.pt',  # model2.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.25,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        trace = True
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights1, device=device, dnn=dnn, data=data, fp16=half)
    model2 = attempt_load(weights2,map_location=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    if half:
        model2.half()
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    names2 = model2.module.names if hasattr(model2, 'module') else model2.names
    colors2 = [[random.randint(0, 255) for _ in range(3)] for _ in names2]
    vid_path, vid_writer = [None] * bs, [None] * bs
    # flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    if device.type != 'cpu':
        model2(torch.zeros(1, 3, imgsz[0], imgsz[0]).to(device).type_as(next(model2.parameters())))
        # model2(torch.zeros(1, model2.yaml.get('ch', 3), imgsz[0], imgsz[0], device=next(model2.parameters()).device))
    old_img_w = old_img_h = imgsz[0]
    old_img_b = 1
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        tracks = False
        # poly=[]
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            # im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im = im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            
            if device.type != 'cpu' and (old_img_b != im.shape[0] or old_img_h != im.shape[2] or old_img_w != im.shape[3]):
                old_img_b = im.shape[0]
                old_img_h = im.shape[2]
                old_img_w = im.shape[3]
                for i in range(3):
                    model2(im, augment=opt.augment)[0]

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred, out = model(im, augment=augment, visualize=visualize)
                
                proto = out[1]
                pred2 = model2(im, augment=opt.augment)[0]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
            pred2 = non_max_suppression_det(pred2, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # pred2 = non_max_suppression_kpt(pred2, 0.5, opt.iou_thres,nc=model2.yaml['nc'], # Number of Classes
            #                         nkpt=model2.yaml['nkpt'], # Number of Keypoints
            #                         kpt_label=True)
            # pred2 = output_to_keypoint(pred2)

    
        # Process predictions
        previous_polygon = []
        polygons = [[]]
        itr = 0
        for i1, det1 in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i1], im0s[i1].copy(), dataset.count
                s += f'{i1}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det1):
                # print('pred:',pred)

                masks = process_mask(proto[i1], det1[:, 6:], det1[:, :4], im.shape[2:], upsample=True)  # HWC
                
                # polygons = []
                if (len(masks)):
                    tracks=True
                    for mask in masks:
                        itr = itr + 1
                        # mask = cv2.resize((mask).detach().cpu().numpy(), (im0.shape[0],im0.shape[1]))
                        scaled = scale_masks(im.shape[2:], mask.detach().cpu().numpy(), im0.shape)
                        contours, _ = cv2.findContours((scaled > 0.7).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            contour = contour.squeeze()
                            poly = contour.tolist()
                            polygons.append(poly)
                            # previous_polygon.append(poly)
                            # area_px = cv2.contourArea(contour)
                            print('len before:',len(polygons[itr]))

                        # polygon_shape = Polygon(polygons[0])
                        # new_poly = simplify_by_angle(polygon_shape)
                        # print('len after: ',len(new_poly))
                        # print(new_poly)

                
                try:
                    # if area_px>6000:
                        # red_poly = reduce_polygon(np.array(polygons))
                    tracks=True
                    polygon_shape = Polygon(polygons[itr])
                    new_poly = simplify_by_angle(polygon_shape)
                    print('len after: ',len(new_poly))
                    print(new_poly)
                    areas = [area(new_poly)]
                    
                    # else:
                    #     tracks=False
                except:
                    # if area_px>6000:
                        # red_poly = reduce_polygon(np.array(polygons))
                    tracks=True
                    polygon_shape = Polygon(polygons[itr-1])
                    new_poly = simplify_by_angle(polygon_shape)
                    print('len after: ',len(new_poly))
                    print(new_poly)
                    areas = [area(new_poly)]
                    
                    # else:
                    #     tracks=False
                    # continue
                    # polygon_shape = Polygon(previous_polygon[0])
                    # new_poly = simplify_by_angle(polygon_shape)
                    # areas = [area(previous_polygon[0])]
                # if area_px>6000:
                #         red_poly = reduce_polygon(np.array(polygons))
                #         tracks=True
                #         areas = [area(red_poly.tolist())]
                # else:
                #     tracks=False
            
                     
                # else:
                #     tracks=False
    

                
                det1[:, :4] = scale_coords(im.shape[2:], det1[:, :4], im0.shape).round()
                for c in det1[:, 5].unique():
                        n = (det1[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det1[:, 5]]
                im_masks = plot_masks(im[i1], masks, mcolors)  # image with masks shape(imh,imw,3)
                # print(polygons)
                
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
            
        im0 = annotator.result()
            
        for i2,det2 in enumerate(pred2):
            seen += 1
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            if len(det2):
                # Rescale boxes from img_size to im0 size
                det2[:, :4] = scale_coords(im.shape[2:], det2[:, :4], im0.shape).round()
                for c in det2[:, -1].unique():
                    n = (det2[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names2[int(c)]}{'s' * (n > 1)}, "  # add to string
            
                for *xyxy, conf, cls in reversed(det2):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        label2 = f'{names2[int(cls)]} {conf:.2f}'
                        # nimg = im[0].permute(1, 2, 0) * 255
                        # nimg = nimg.cpu().numpy().astype(np.uint8)
                        # nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
                        # for idx in range(pred2.shape[0]):
                        #     plot_skeleton_kpts(nimg, pred2[idx, 7:].T, 3)
                        plot_one_box(xyxy, im0, label=label2, color=colors2[int(cls)], line_thickness=1)

                        bot_r=(int(xyxy[2]), int(xyxy[3]))
                        if tracks==True:
                            if (point_intersection_shapely(polygons[0],[bot_r])==True) :
                                cv2.putText(im0,"Detected",(100,100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,3,(255,0,0), thickness=8,lineType=cv2.LINE_AA)
            # Stream results
        # intersection_detected = check_intersection(person_xyxy, scaled_mask)
        if tracks==True:
            drawAreas(im0,areas)
        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(21)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det1) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights1[0])
        strip_optimizer(weights2[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights1', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--weights2', nargs='+', type=str, default=ROOT / 'yolov7.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
