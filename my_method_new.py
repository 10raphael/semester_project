import torch
from torchvision import transforms
from torchvision.models.detection import fcos_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2, keypointrcnn_resnet50_fpn, ssd300_vgg16

from PIL import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

import time
import os

'''
Part of my semester project "Object Instance Segmentation for 3D Reconstruction"
this is the script used to generate the input for XMem, saved in /output 

the code is structured as follows:
- process_frame (outputs the input for SAM)
    - filter_big_boxes
    - sort_box2center
    - cand_human
    - find_human
- make_sam_input
- get_mask (returns mask)
- find_biggest_mask (which mask to be used for SAM)

beyond the official documentation, links to various sources are provided where I used them, used chatgpt for debugging
'''

# adjust img_dir to point to the directory where the input sequence (rgb) is saved
img_dir = '/local/home/geraphae/sem_proj/my_BEHAVE/04_tablesquare_lift.3/rgb_raw'
# img_dir = '/local/home/geraphae/sem_proj/my_BEHAVE/05_yogamat.3/rgb_raw'
# img_dir = '/local/home/geraphae/sem_proj/my_BEHAVE/05_boxlarge.1_mini/rgb_raw'

# only keep the bounding boxes of a certain minimal size
def filter_big_boxes(prediction, min_size=180):
    # print(prediction[0])

    # prediction =[boxes, labels, scores, masks]

    big_boxes = [
        # box for box in prediction[0]['boxes'] if
        box for box in prediction[0] if
        box[2] - box[0] > min_size and
        box[3] - box[1] > min_size
    ]

    # print('big_boxes:', big_boxes)
    # print(type(prediction), type(big_boxes))

    return big_boxes

# sort boxes by distance to the center of the frame, closest one first
def sort_box2center(big_boxes, img):

    # https://codoraven.com/tutorials/opencv-vs-pillow/image-shape-size/
    # W, H = img.size
    H, W, _ = img.shape
    # print('W', W)
    img_center = (W/2, H/2)
    
    # print(img_center)
    # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    dist2center = [np.linalg.norm([((box[0] + box[2])/2) - img_center[0], ((box[1] + box[3])/2) - img_center[1]]) for box in big_boxes]
    # print(dist2center)

    dist2center_float = [float(dist) for dist in dist2center]

    sorted_big_boxes = [box for box, _ in sorted(zip(big_boxes, dist2center_float), key=lambda x: x[1])]
    return sorted_big_boxes

# first box of sort_bigbox list as nparray and in same format as human detector
def cand_human(sorted_big_boxes):
    # print(sorted_big_boxes[0])
    cur_best_box = [sorted_big_boxes[0][0], sorted_big_boxes[0][1], sorted_big_boxes[0][2] - sorted_big_boxes[0][0], sorted_big_boxes[0][3] - sorted_big_boxes[0][1]]
    return np.array(cur_best_box)

# https://thedatafrog.com/en/articles/human-detection-video/
# https://stackoverflow.com/questions/76667072/how-to-use-custom-svm-detector-with-cv2-hogdescriptor
# https://stackoverflow.com/questions/2188646/how-can-i-detect-and-track-people-using-opencv

# returns a prdicted bounding box around the human
def find_human(img_cv):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    human, _ = hog.detectMultiScale(img_cv, winStride=(8, 8))

    # only one human in frame, wrong if more detected
    if len(human) > 1:
        human = None
    
    # print(type(human), human)

    return human

# one frame, returns input for SAM
def process_frame(img):
    # print(img)
    model.eval()
    # https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
    # img = Image.open(img_path)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
 
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # prediction = [boxes, labels, scores, masks]
    # https://stackoverflow.com/a/77260393
    pred_boxes = [pred['boxes'].cpu().numpy() for pred in prediction] 
    # print(type(prediction), prediction)

    big_boxes = filter_big_boxes(pred_boxes)
    
    if big_boxes is None:
        print('no object found?')
        return None
    
    sorted_big_boxes = sort_box2center(big_boxes, img)

    cur_box = (cand_human(sorted_big_boxes))
    # print(type(cur_box), 'cur_box')

    human = find_human(img)

    # discard the box if a human might be in it
    if human is not None and len(human) > 0:
        # there must be at least 2 boxes, whithin 200pxl of the detected human
        if len(sorted_big_boxes) >= 2:
            if np.all(np.abs(cur_box - human) <= 200):
                print('human selected, pick next closest box')

                sorted_big_boxes = sorted_big_boxes[1:]
        else:
            print('only human found')

    # sanity check
    elif cur_box[3] > (2.5 * cur_box[2]) and sorted_big_boxes:
        if len(sorted_big_boxes) >= 2:
            print('probably a human, pick next best box to be safe')

            sorted_big_boxes = sorted_big_boxes[1:]
    else:
        print('only one box found')

    box_now = sorted_big_boxes[0]
    # selected box on top of array
    # print(type(box_now))

    box_sam = np.array(box_now)
    # print(box_sam)

    return box_sam

# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# provided utils from SAM github
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
# provided util    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
# provided util    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    # print(f"show_box Box: {list(box)}")
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# select the midpoint of the box and make it the input for SAM
def make_sam_input(box):
    # point prompt : star on midpoint of box
    input_point = np.array([[0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])]])
    input_label = np.array([1])  # 1: green, 0: red(avoid)
    return input_point, input_label


def display_input(img, box, input_point, input_label, model_name):
    plt.imshow(img)
    show_box(box, plt.gca())
    show_points(input_point, input_label, plt.gca())
    # show_mask(mask, plot.gca())
    plt.axis('off')
    plt.title('SAM input, ' + str(model_name))
    plt.show()

# SAM used here, produces mask
def get_mask(predictor, img, input_point, input_label, box_sam, out_dir):
    
    predictor.set_image(img)
    
    # https://github.com/facebookresearch/segment-anything/tree/main#getting-started
    # multimask true outputs 3 masks
    masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label, box=None, multimask_output=False)
    # print('estimated IoU:', scores[0])
    
    # max_score = np.argmax(scores)
    # mask = masks[max_score]

    # show_box(box_sam, plt.gca())
    # show_mask(masks[0], plt.gca())
    # show_points(input_point, input_label, plt.gca())
    
    # https://stackoverflow.com/questions/56942102/how-to-generate-a-mask-using-pillows-image-load-function
    mask_path = os.path.join(out_dir, f'{cur_frame}')
    Image.fromarray(masks[0]).save(mask_path)
    # Image.fromarray(mask).save(mask_path)

# "post-processing": out of the masks made by SAM, select the one with the most white pixels
def find_biggest_mask(in_path):
    list_masks = os.listdir(in_path)
    masks = sorted([mask for mask in list_masks])

    best_mask = None
    most_pxl = 0

    for mask_idx in masks:
        
        mask_path = os.path.join(in_path, mask_idx)
        
        # https://stackoverflow.com/questions/47494350/count-total-number-of-white-pixels-in-an-image
        # white is 255 in opencv, 1 in pil
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        num_pxl_white = np.sum(mask == 255)
        # print('# of pixels in mask:', num_pxl_white)

        if num_pxl_white > most_pxl:
            most_pxl = num_pxl_white
            best_mask = mask_idx

    # best_mask.show()
    print(f'mask for XMem input: {out_dir}/{best_mask}')
    # return best_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device} for inference')

# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html
# https://pytorch.org/vision/master/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html
# model = fcos_resnet50_fpn(weights='DEFAULT', pretrained=True).to(device)
# model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', pretrained=True).to(device)
model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT', pretrained=True).to(device)

model_name = type(model).__name__
print('object detector used:', model_name)
'''
img_dir = '/local/home/geraphae/sem_proj/my_BEHAVE/04_tablesquare_lift.3/rgb_raw'
# img_dir = '/local/home/geraphae/sem_proj/my_BEHAVE/05_yogamat.3/rgb_raw'
# img_dir = '/local/home/geraphae/sem_proj/my_BEHAVE/05_boxlarge.1_mini/rgb_raw'
'''
in_dir = sorted(os.listdir(img_dir))

out_new = 'output'

if not os.path.exists(out_new):
    os.makedirs(out_new)

seq = os.path.basename(os.path.dirname(img_dir))
print("sequence:", seq)

# https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device)
predictor = SamPredictor(sam)

t_det = []
t_seg = []

for cur_frame in in_dir[:10]:

    img_path = os.path.join(img_dir, cur_frame)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    print('processing frame no.:', cur_frame)

    t_start = time.time()
    box = process_frame(img)
    
    t_detect = time.time()-t_start
    # print('time for object detection:',round(t_detect, 2),'s')
    # print('Box for SAM input', box)
    t_det.append(t_detect)

    input_point, input_label = make_sam_input(box)
    
    out_dir = os.path.join(out_new, seq, model_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # display_input(img, box, input_point, input_label, model_name)

    t_start = time.time()
    get_mask(predictor, img, input_point, input_label, box, out_dir)
    # print(out_dir)
    
    t_segm = time.time()-t_start
    t_seg.append(t_segm)
    # print('time for segmentation:',round(t_segm, 2),'s')

    print(f'mask {cur_frame} saved')

print('AVG time for object detection:', round(np.mean(t_det), 2),'s')
print('AVG time for segmentation:', round(np.mean(t_seg), 2),'s')

find_biggest_mask(out_dir)

print('DONE')
