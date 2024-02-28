import torch
import cv2
import numpy as np
import mmcv
import mmengine
from mmengine.structures import InstanceData
from mmdet.structures.bbox import scale_boxes, get_box_wh
from ensemble_boxes import *


def is_night(labels, scores, entities):
    night_flag = False
    day_falg = False
    
    for label, score in zip(labels, scores):   # TODO day, night의 box 영역 비교해서 판단
        s = entities[label]
        if s == 'night':
            night_flag = True
        if s == 'day':
            day_flag = True
    
        if night_flag and not day_flag:
            return True
        
        return False

def select_main_label(box_list, labels, scores):
    # 각 박스의 중복을 제거하고 가장 높은 점수를 가진 라벨만 남김
    selected_boxes = []
    selected_scores = []
    selected_labels = []
    unique_boxes = set(tuple(box) for box in box_list)
    for unique_box in unique_boxes:
        max_score = 0
        max_label = None
        for box, score, label in zip(box_list, scores, labels):
            if tuple(box) == unique_box:
                if score > max_score:
                    max_score = score
                    max_label = label
        selected_boxes.append(list(unique_box))
        selected_scores.append(max_score)
        selected_labels.append(max_label)

    return selected_boxes, selected_labels, selected_scores

def remove_small_box(box_list, labels, scores, entities):
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    # TODO _bbox_post_process에서 최소 크기 정해서 thresholding 가능
    for box, score, label_id in zip(box_list, scores, labels):
        # w, h = get_box_wh(box)
        w = box[2] - box[0]
        h = box[3] - box[1]
        label = entities[label_id]
        
        # TODO 크기 대신 normalize한 뒤의 값으로 threshold?
        if label in ["heels", "sneakers", "shoes"] :
            if w > 70 and h > 70:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label_id)
        elif label in ["bag", "wallet", "backpack", "suitcase"]:
            if w > 30 and h > 30:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label_id)
        elif label in ["watch", "necklace", "bracelet", "ring", "earrings"]:
            if w > 30 and h > 30:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label_id)
        else :
            # print("Leave apparel boxes if w> 100 and h> 100")
            if w > 100 and h > 100:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label_id)        
    return filtered_boxes, filtered_labels, filtered_scores

def check_face_area_ratio(box_list, labels, scores, entities, img_size):
    imgh, imgw =  img_size
    img_area = imgh * imgw
    max_area = 0.0
    for box, label, score in zip(box_list, labels, scores):
        s = entities[label] 
        if s == 'face':
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if max_area < area:
                max_area = area

    ratio = 100*max_area/img_area
    print("****>> face ratio : {0}".format(ratio))
    return ratio
    
def filter_zoomin_image(box_list, labels, scores, entities, img_size, face_threshold=10):
    if len(labels)==0:
        return box_list, labels, scores
    else:
        ratio = check_face_area_ratio(box_list, labels, scores, entities, img_size)
        
        if ratio > face_threshold:  # for zoomed-in images
            new_boxes = []
            new_labels = []
            new_scores = []
            for box, label, score in zip(box_list, labels, scores):
                s = entities[label] 
                if s in ['watch', 'necklace', 'bracelet', 'ring', 'earrings']:
                    new_boxes.append(box)
                    new_labels.append(label)
                    new_scores.append(score)
            return new_boxes, new_labels, new_scores
            
        else:
            return box_list, labels, scores

def integrate_subclass(labels, entities):
    new_labels = []
    for label_id in labels:
        s = entities[label_id]
        
        if s in ['t-shirt', 'shirt', 'sweater', 'coat']:
            new_ID = entities.index('t-shirt')
        elif s in ['dress', 'pants', 'skirt']:
            new_ID = entities.index('dress')
        elif s in ['heels', 'sneakers', 'shoes']:
            new_ID = entities.index('shoes')
        elif s in ['bag', 'wallet', 'backpack', 'suitcase']:
            new_ID = entities.index('bag')
        else:
            new_ID = label_id
        new_labels.append(new_ID)
    return new_labels

def euclideandist(a, b):
    #dist = np.sqrt(np.sum(np.square(a - b)))

    dist = np.linalg.norm(a - b)
    return dist

def find_smilarycenter_box(srcbox, targetbox): 

    maxsim_index_list = []

    # print("*****> srcbox: {0}".format(srcbox))
    # print("*****> targetbox: {0}".format(targetbox))

    for sc in srcbox:
        min_dist = 1000000000.0 
        min_index = -1
        index = 0
        npsc = np.array(sc)
        for tc in targetbox:
            #sim = self.cosine_similarity(sc, tc)
            nptc = np.array(tc)
            
            dist = euclideandist(npsc, nptc)
            if min_dist > dist:
                min_dist = dist
                min_index = index
            index = index +1

        maxsim_index_list.append(min_index)    

        # print("min_index :{0}, min dist:{1}".format(min_index, min_index))
    return maxsim_index_list

def apply_nms(box_list, labels, scores, entities, img_size):
    if len(labels)>1:
        # normalize in range [0:1]
        h, w =  img_size
        scale_factor = [1/w, 1/h]
        boxes_normalized = scale_boxes(torch.tensor(box_list, dtype=torch.float32), scale_factor)
        
        # 중분류 단위로 묶어서 nms 진행
        integrated_labels = integrate_subclass(labels, entities)
        
        # for nms
        weights = [2, 1]
        iou_thr = 0.5
        skip_box_thr = 0.0001
        sigma = 0.1
        
        # for single model predictions
        boxes_nms, scores_nms, labels_nms = soft_nms([boxes_normalized], [scores], [integrated_labels], weights=None, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        
        # boxes2_f, scores2, labels2_f = nms([box_list], [scores], [new_labels], weights=None, iou_thr=iou_thr)
        # boxes2_f, scores2, labels2_f = non_maximum_weighted([box_list], [scores], [new_labels], weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        # boxes2_f, scores2, labels2_f = weighted_boxes_fusion([box_list], [scores], [new_labels], weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        
        maxsim_index_list = find_smilarycenter_box(boxes_nms, boxes_normalized)
        
        recover_labels_list = []
        for index in maxsim_index_list:
            label = labels[index]
            recover_labels_list.append(label)
        
        # resize to original size
        scale_factor = [w, h]
        boxes_final = scale_boxes(torch.tensor(boxes_nms, dtype=torch.float32), scale_factor)
        
        return boxes_final.tolist(), recover_labels_list, scores_nms
    else:
        return box_list, labels, scores

def check_in_box(box_list, labels, scores, entities, target_index, check_labels, margin=10):
    deleteindex_list = []
    # target_indices = [i for i, label in enumerate(labels) if entities[label] in target_labels]
    check_indices = [i for i, label in enumerate(labels) if entities[label] in check_labels]
    
    # for tindex in target_indices:
    target_box = box_list[target_index]
    target_score = scores[target_index]
    dx1, dy1, dx2, dy2 = target_box[0]-margin, target_box[1]-margin, target_box[2]+margin, target_box[3]+margin
    max_check_score = -1
    sub_indices = []
    for cindex in check_indices:
        x1, y1, x2, y2 = box_list[cindex]
        if dx1 <= x1 and dy1 <= y1 and dx2 >= x2 and dy2 >= y2:
            check_score = scores[cindex] 
            if check_score > max_check_score:
                max_check_score = check_score   # 내부 box 중 최대 score
                sub_indices.append(cindex)
    
    if max_check_score > target_score:   # delete target box
        deleteindex_list.append(target_index)
    else:   # delete sub box
        deleteindex_list.extend(sub_indices)
            
    return deleteindex_list       

def check_boxes(box_list, labels, scores, entities):
    if len(labels)==0:
        return box_list, labels, scores
    else:
        delete_index_list = []
        for tindex, (box, label, score) in enumerate(zip(box_list, labels, scores)):
            s = entities[label]
            
            if s == 'dress':
                dress_delete_indices = check_in_box(box_list, labels, scores, entities, tindex, ['shirt', 't-shirt', 'pants', 'skirt'])
                delete_index_list.extend(dress_delete_indices)
            
            if s =='suit':
                suit_delete_indices = check_in_box(box_list, labels, scores, entities, tindex, ['shirt', 't-shirt', 'sweater', 'coat'])
                delete_index_list.extend(suit_delete_indices)
                
            if s in ['heels', 'sneakers', 'shoes']:
                shoes_delete_indices = check_in_box(box_list, labels, scores, entities, tindex, ['watch', 'necklace', 'bracelet', 'ring', 'earrings'])
                delete_index_list.extend(shoes_delete_indices)
                
            if s in ['necklace', 'bag'] and score < 0.6:
                delete_index_list.append(tindex)
                
            if s in ['face', 'day', 'night']:
                delete_index_list.append(tindex)
        
        new_boxes = []
        new_labels = []
        new_scores = []

        for i, (box, label , score) in enumerate(zip(box_list, labels, scores)):
            try:
                index = delete_index_list.index(i)   # i가 deleteindex_list에 포함되는지 확인
            except ValueError:  
                #if index == -1 :
                new_boxes.append(box)
                new_labels.append(label)
                new_scores.append(score)
        
        return new_boxes, new_labels, new_scores

def check_tops(box_list, labels, scores, entities, margin=10):
    if len(labels)==0:
        return box_list, labels, scores
    else:
        target_labels = ['shirt', 't-shirt', 'sweater', 'coat', 'suit']
        check_labels = ['shirt', 't-shirt', 'sweater', 'dress']
        
        target_indices = [i for i, label in enumerate(labels) if entities[label] in target_labels]
        check_indices = [i for i, label in enumerate(labels) if entities[label] in check_labels]
        
        delete_index_list = []
        for tindex in target_indices:
            target_box = box_list[tindex]
            target_score = scores[tindex]
            dx1, dy1, dx2, dy2 = target_box[0]-margin, target_box[1]-margin, target_box[2]+margin, target_box[3]+margin
            target_area = (dx2 - dx1) * (dy2 - dy1)
            
            for cindex in check_indices:
                x1, y1, x2, y2 = box_list[cindex]
                if dx1 <= x1 and dy1 <= y1 and dx2 >= x2 and dy2 >= y2:
                    check_area = (x2 - x1) * (y2 - y1)
                    ratio = check_area / target_area
                    
                    if ratio < 0.6:
                        # scores[i] = scores[i]*0.5
                        delete_index_list.append(cindex)

        new_boxes = []
        new_labels = []
        new_scores = []

        for i, (box, label , score) in enumerate(zip(box_list, labels, scores)):
            try:
                index = delete_index_list.index(i)   # i가 deleteindex_list에 포함되는지 확인
            except ValueError:  
                #if index == -1 :
                new_boxes.append(box)
                new_labels.append(label)
                new_scores.append(score)
        
        return new_boxes, new_labels, new_scores

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance o            f the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()
    
def isit_blurringimage(image, threshold=70.0):
    #convert PIL to cv2 image
    # use numpy to convert the pil_image into a numpy array
    #numpy_image=np.array(pil_image)  

    # convert to a openCV2 image and convert from RGB to BGR format
    #opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    opencv_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(opencv_image)
    print("blurring score:{}".format(fm))
    
    text = "Not Blurry"
    blurring_flag = False
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < threshold:
        text = "Blurry"
        print("blurring image : {0}".format(fm))
        blurring_flag = True

    return blurring_flag

def remove_blurred_box(box_list, labels, scores, image_path):
    if len(labels)==0:
        return box_list, labels, scores
    else:
        img_bytes = mmengine.fileio.get(image_path)
        img = mmcv.imfrombytes(img_bytes)
        
        new_boxes = []
        new_labels = []
        new_scores = []

        for box, label, score in zip(box_list, labels, scores):
            # x1, y1, x2, y2 = box
            x1, y1, x2, y2 = np.array(box, dtype=np.int64)
            if x1 < x2 and y1 < y2:
                crop_image = img[y1:y2, x1:x2, ::-1]
                if not isit_blurringimage(crop_image, threshold=20.0):
                    new_boxes.append(box)
                    new_labels.append(label)
                    new_scores.append(score)
        return new_boxes, new_labels, new_scores