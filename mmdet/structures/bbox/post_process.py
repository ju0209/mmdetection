import torch
import numpy as np
from mmengine.structures import InstanceData
from mmdet.structures.bbox import scale_boxes, get_box_wh
from ensemble_boxes import *

    
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
    
    if len(filtered_boxes)==0:
        filtered_boxes = torch.empty(0,4)

    return filtered_boxes, filtered_labels, filtered_scores

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

def apply_nms(data_sample, box_list, labels, scores, entities):
    if len(labels)>1:
        
        # normalize in range [0:1]
        h, w =  data_sample.img_shape
        scale_factor = [1/w, 1/h]
        boxes_normalized = scale_boxes(torch.tensor(box_list, dtype=torch.float32), scale_factor)
        
        # 중분류 단위로 묶어서 nms 진행
        integrated_labels = integrate_subclass(labels, entities)
        
        # for nms
        weights = [2, 1]
        iou_thr = 0.5
        skip_box_thr = 0.0001
        sigma = 0.1
        
        #for single model predictions
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
    
# def add_label_names(data_sample, labels, entities):
#     if len(labels) > 0:
#         label_names = []
#         for label in labels:
#             if labels >= len(entities):
#                 warnings.warn(
#                     'The unexpected output indicates an issue with '
#                     'named entity recognition. You can try '
#                     'setting custom_entities=True and running '
#                     'again to see if it helps.')
#                 label_names.append('unobject')
#             else:
#                 label_names.append(entities[label])
        

def check_dress(box_list, labels, scores, entities, margin=10):
    if len(labels)==0:
        return box_list, labels, scores
    else:
        deleteindex_list = []
        dindex = 0
        for box, label, score in zip(box_list, labels, scores):
            s = entities[label]
            if s == 'dress':
                dx1, dy1, dx2, dy2 = box[0]-margin, box[1]-margin, box[2]+margin, box[3]+margin
                index = 0
                for box2, label2, score2 in zip(box_list, labels, scores):
                    s2 = entities[label2] 
                    if s2 == 'shirt' or s2 == 'pants' or s2 == 'skirt':
                        x1, y1, x2, y2 = box2
                        if dx1 <= x1 and dy1 <= y1 and dx2 >= x2 and dy2 >= y2:
                            if score >= 0.6:    # delete sub box
                                deleteindex_list.append(index)
                            else:   # delete dress
                                deleteindex_list.append(dindex)
                    index = index +1     
            dindex = dindex + 1
        
        new_boxes = []
        new_labels = []
        new_scores = []

        for i, (box, label , score) in enumerate(zip(box_list, labels, scores)):
            try:
                index = deleteindex_list.index(i)   # i가 deleteindex_list에 포함되는지 확인
            except ValueError:  
                #if index == -1 :
                new_boxes.append(box)
                new_labels.append(label)
                new_scores.append(score)
                
        return new_boxes, new_labels, new_scores
    
def check_shoes(box_list, labels, scores, entities):
    if len(labels)==0:
        return box_list, labels, scores
    else:
        deleteindex_list = []
        dindex = 0
        for box, label, score in zip(box_list, labels, scores):
            s = entities[label]
            
            if s in ["heels", "sneakers", "shoes"]:
                dx1, dy1, dx2, dy2 = box[0], box[1], box[2], box[3]
                index = 0
                for box2, label2, score2 in zip(box_list, labels, scores):
                    s2 = entities[label2] 
                    if s2 != 'shoes' and s2 != 'heels' and s2 != 'sneakers':
                        x1, y1, x2, y2 = box2
                        if dx1 <= x1 and dy1 <= y1 and dx2 >= x2 and dy2 >= y2:
                            deleteindex_list.append(index)
                    index = index +1
            dindex = dindex + 1      
        new_boxes = []
        new_labels = []
        new_scores = []

        for i, (box, label , score) in enumerate(zip(box_list, labels, scores)):
            try:
                index = deleteindex_list.index(i)
            except ValueError:
                #if index == -1 :
                new_boxes.append(box)
                new_labels.append(label)
                new_scores.append(score)
                
        return new_boxes, new_labels, new_scores