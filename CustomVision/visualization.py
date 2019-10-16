#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
Module documentation:

Project           : Electric and water meter detection
Program name      : visualisation.py
Author            : Bjarne Meyn 
Date created      : 20191016
description       : Collection of visualization functions

'''

def plot_image_detection(image_name, test_images_path, best_box, gt_box):

    img = cv2.imread(test_images_path + image_name)

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    img_height, img_width,_ = img.shape

    pixel_box, pixel_box_gt = calculate_pixel_box(best_box,gt_box, test_images_path, image_name)


    rect = patches.Rectangle((pixel_box[0], pixel_box[1]), pixel_box[2],pixel_box[3], linewidth=2, label="Detection", edgecolor="r", fill=False)
    rect_gt = patches.Rectangle((pixel_box_gt[0], pixel_box_gt[1]), pixel_box_gt[2],pixel_box_gt[3], linewidth=2, label="Groundtruth", edgecolor="g", fill=False)
    
    ax.add_patch(rect)
    ax.add_patch(rect_gt)
    ax.legend()
    plt.show()