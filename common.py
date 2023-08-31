'''
all common code/global strings for the various files for scanned documents
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import layoutparser as lp

# Utility Functions

def union_box(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return [x1, y1, x2, y2]

def intersect_box(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """

    
    if box1[0] > box2[0]:
        tbox = box1
        box1 = box2
        box2 = tbox
        
    if box1[3] < box2[1] or box1[2] < box2[0] or box1[1] > box2[3] or box1[0] > box2[2]:
        return [0, 0, 0, 0]
        
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    return [x1, y1, x2, y2]

def box_area(box):
    return int((box[2]-box[0])*(box[3]-box[1]))

def is_same_line(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    
    box1_midy = (box1[1] + box1[3]) / 2
    box2_midy = (box2[1] + box2[3]) / 2

    if box1_midy < box2[3] and box1_midy > box2[1] and box2_midy < box1[3] and box2_midy > box1[1]:
        return True
    else:
        return False
    
def is_adj_line(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    h1 = box1[3] - box1[1]
    h2 = box2[3] - box2[1]
    h_dist = max(box1[1], box2[1]) - min(box1[3], box2[3])

    box1_midx = (box1[0] + box1[2]) / 2
    box2_midx = (box2[0] + box2[2]) / 2

    # if h_dist <= min(h1, h2) and box1_midx < box2[2] and box1_midx > box2[0] and box2_midx < box1[2] and box2_midx > box1[0]:
    if h_dist <= min(h1, h2): # v2
        return True
    else:
        return False
    
def boxes_sort(boxes):
    """ From left top to right bottom
    Params:
        boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    """
    sorted_id = sorted(range(len(boxes)), key=lambda x: (boxes[x][1], boxes[x][0]))

    # sorted_boxes = [boxes[id] for id in sorted_id]


    return sorted_id

def intersetion_area_ratio_score(box1, box2):
    box1_area = box_area(box1)
    box2_area = box_area(box2)
    min_box_area = min(box1_area, box2_area)
    return box_area(intersect_box(box1, box2))/min_box_area 

def space_layout(texts, boxes):
    line_boxes = []
    line_texts = []
    max_line_char_num = 0
    line_width = 0
    # print(f"len_boxes: {len(boxes)}")
    while len(boxes) > 0:
        line_box = [boxes.pop(0)]
        line_text = [texts.pop(0)]
        char_num = len(line_text[-1])
        line_union_box = line_box[-1]
        while len(boxes) > 0 and is_same_line(line_box[-1], boxes[0]):
            line_box.append(boxes.pop(0))
            line_text.append(texts.pop(0))
            char_num += len(line_text[-1])
            line_union_box = union_box(line_union_box, line_box[-1])
        line_boxes.append(line_box)
        line_texts.append(line_text)
        if char_num >= max_line_char_num:
            max_line_char_num = char_num
            line_width = line_union_box[2] - line_union_box[0]
    
    # print(line_width)
    
    if max_line_char_num ==0:
        char_width = 1
    else:
        char_width = line_width / max_line_char_num
    #     print(char_width)
        if char_width == 0:
            char_width = 1

    space_line_texts = []
    for i, line_box in enumerate(line_boxes):
        space_line_text = ""
        for j, box in enumerate(line_box):
            left_char_num = int(box[0] / char_width)
            space_line_text += " " * (left_char_num - len(space_line_text))
            space_line_text += line_texts[i][j]
        space_line_texts.append(space_line_text)

    return space_line_texts

def convert_box4point_to_box2point(box4point):
    all_x = [int(box4point[i][0]) for i in range(4)]
    all_y = [int(box4point[i][1]) for i in range(4)]
    box2point = [min(all_x), min(all_y), max(all_x), max(all_y)]
    return box2point

def get_text_bbox_from_sample(sample_image, reader, box_num_threshold = 30):
    
    texts = []
    boxes = []
    confidence_scores = []
    
    result = reader.ocr(sample_image, cls=False)[0]

    if len(result) <= box_num_threshold:
        return [], [], []

    for r in result:
        texts.append(r[1][0])
        confidence_scores.append(r[1][1])
        boxes.append(convert_box4point_to_box2point(r[0]))
        
    return texts, boxes, confidence_scores
    
def separate_text_image_boxes(texts, boxes, image_boxes, intersection_threshold):
    only_text, image_text, only_text_box, image_text_box = [], [], [], []
    
    for i in range(len(boxes)):
        is_image_text = False
        
        for j in range(len(image_boxes)):
            intersetion_score = intersetion_area_ratio_score(boxes[i], image_boxes[j])
#             print(f"intersetion_score: {intersetion_score}")
            if intersetion_score > intersection_threshold:
                image_text.append(texts[i])
                image_text_box.append(boxes[i])
                is_image_text = True
                break
                
        if is_image_text == False:
            only_text.append(texts[i])
            only_text_box.append(boxes[i])

    return [only_text, only_text_box], [image_text, image_text_box]

def get_text_in_each_image(sample_image, image_boxes, reader):
    img_text_list = []
    for box in image_boxes:
        timg = np.array(sample_image[box[1]:box[3], box[0]: box[2],:])
#         plt.imshow(timg)
#         plt.show()
        result = reader.readtext(timg)
        img_text = []
        for r in result:
            img_text.append(r[1])
        img_text = " ".join(img_text)
#         print(f"img text {img_text}")
        img_text_list.append(img_text)
        
    return img_text_list

def check_image_box_intersection(image_boxes):
    print("\n\n\n\n\n\n -----------")
    for i in range(len(image_boxes)):
        for j in range(len(image_boxes)):
            if i==j:
                continue
            intersection_score = intersetion_area_ratio_score(image_boxes[i], image_boxes[j])
            if intersection_score > 0:
                print(f"box {i} intersection box {j}: {intersection_score}")
                
    print("\n ----------- \n\n\n\n\n\n")

def visualize_text_boxes(sample_image, boxes):
    sample_image = sample_image.copy()
    for box in boxes:
        cv2.rectangle(sample_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        
    plt.imshow(sample_image)
    plt.show()
    
'''
caveat numpy inmt does not work with cv2, 
for shapes convert coordinates to python int
also you need to make a copy of the image array for this work 
god knows why
'''

def merge_text_boxes(text_boxes):
    merged_text_boxes = []
    tboxes = text_boxes
    while True:
        did_merge = False
        for i in range(len(tboxes)):
            if tboxes[i] == [0,0,0,0]:
                continue
            for j in range(len(tboxes)):
                if i == j or tboxes[j]==[0,0,0,0]:
                    continue
                elif intersetion_area_ratio_score(tboxes[i], tboxes[j])> 0.5:
                    print(f"iscore {intersetion_area_ratio_score(tboxes[i], tboxes[j])}")
                    tboxes[i] = union_box(tboxes[i], tboxes[j])
                    tboxes[j] = [0,0,0,0]
                    did_merge = True

            merged_text_boxes.append(tboxes[i])

        tboxes = merged_text_boxes
        merged_text_boxes = []
        if did_merge == False:
            break
    
    merged_text_boxes = tboxes
    return merged_text_boxes

# Main function that converts final ocr to description
def convert_sample_to_description(sample_image, reader, box_num_threshold = 0, visualize = False):
    
    texts, text_boxes, confidence_scores = get_text_bbox_from_sample(sample_image, reader, box_num_threshold)
    
    text_boxes = merge_text_boxes(text_boxes)
    ids = boxes_sort(text_boxes)
    text_boxes = [text_boxes[i] for i in ids]
    
    if visualize == True:
        visualize_text_boxes(sample_image, text_boxes)
    
    space_lines_text = space_layout(texts, text_boxes)
    doc_layout_as_text = "\n".join(space_lines_text)
    
    return doc_layout_as_text
