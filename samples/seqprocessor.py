import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import argparse
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


def get_class_names():
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
    return class_names


def prepare_model():
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model


def infer_single_image(img_path, model):


    image = skimage.io.imread(img_path)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

    w = image.shape[1]
    h = image.shape[0]

    return r, w, h


def convert_output(r, w, h, default_class_label_orig):
    n = len(r['scores'])
    if (n == 0):
        max_scores = np.zeros((h, w))
        instance_index = np.zeros((h, w), dtype=np.uint8)
        class_labels_orig = np.ones((h, w), dtype=np.uint8) * default_class_label_orig
        return max_scores, instance_index, class_labels_orig
    score_mask = np.array(r['scores']).reshape(1, 1, -1)
    score_mask = np.tile(score_mask, (h, w, 1))

    final_score_mask = np.zeros((h, w, n))
    final_score_mask[r['masks']] = score_mask[r['masks']]
    instance_index = np.argmax(final_score_mask, axis=2)
    max_scores = np.max(final_score_mask, axis=2)

    class_labels_orig = np.ones((h, w), dtype=np.uint8) * default_class_label_orig

    class_ids = r['class_ids'] 
    instance_index_1d = instance_index.reshape(-1)
    max_scores_1d = max_scores.reshape(-1)
    m = (max_scores_1d > 0.0)

    class_labels_orig_1d = class_labels_orig.reshape(-1)
    class_labels_orig_1d[m] = class_ids[instance_index_1d[m]]

    # our convention: instance_index == 0 for non-things, instances are indexed starting from 1
    instance_index += 1
    instance_index_1d[~m] = 0

    assert((np.max(np.abs(np.asarray(r['scores'])[instance_index_1d[m]-1] - max_scores.reshape(-1)[m])) < 1e-6) and "Problem with scores")

    max_mask = np.max(r['masks'], axis=2)
    assert((np.max(np.abs(max_mask.reshape(-1).astype(int) - m.astype(int))) == 0) and "Problem with overall mask")
    if (n > 0):
        assert((np.max(np.abs((instance_index == 1).astype(int) - r['masks'][:, :, 0].astype(int))) == 0) and "First mask problem") 

    return max_scores, instance_index, class_labels_orig


def process_sequence(path_in, path_out, model):
    default_class_label_orig = len(get_class_names())
    img_files = sorted(os.listdir(path_in))
    out_labels_orig_path  = path_out + '/labels'
    os.makedirs(out_labels_orig_path, exist_ok=True)
    out_instances_path = path_out + '/instances'
    os.makedirs(out_instances_path, exist_ok=True)
    out_confs_path = path_out + '/confs'
    os.makedirs(out_confs_path, exist_ok=True)
    for img_fname in img_files:
        if 'png' in img_fname:
            print(img_fname)
            r, w, h = infer_single_image(path_in + '/' + img_fname, model)
            max_scores, instance_index, class_labels_orig = convert_output(r, w, h, default_class_label_orig)
            cv2.imwrite(out_labels_orig_path + '/' + img_fname, class_labels_orig)
            cv2.imwrite(out_instances_path + '/' + img_fname, instance_index)
            cv2.imwrite(out_confs_path + '/' + img_fname, (max_scores * 255).astype(np.uint8))


def process_all(model):
    for seq_id in ['016', '030', '061', '078', '086', '096', '206', '223', '255']: # '011' omitted
        prefix = '/mnt/a563050a-fc9d-4558-af28-32fce91f02b8/data/scenenn/103.24.77.34/scenenn/main/oni_unpacked/' + seq_id + '/'
        process_sequence(prefix + 'image/', prefix + '/mask_rcnn', model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.description = "Process a sequence with Mask-RCNN"
    parser.add_argument(
        "-i",
        "--in",
        required=False,
        help="Path to the image folder",
        default=""
    )

    parser.add_argument(
        "-o",
        "--out",
        required=False,
        help="Path to the output folder",
    )

    parser_args = vars(parser.parse_args())

    model = prepare_model()

    if len(parser_args["in"]) == 0:
        process_all(model)
    else:
        process_sequence(parser_args['in'], parser_args['out'], model)