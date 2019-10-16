#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
Module documentation:

Project           : Electric and water meter detection
Program name      : helperfunctions.py
Author            : Bjarne Meyn 
Date created      : 20191016
description       : Collection of functions to preprocess data or convert datatype
'''

# Imports
import cv2

def calculate_pixel_box(best_box, gt_box, test_images_path, image_name):

    img = cv2.imread(test_images_path + image_name)
    img_height, img_width,_ = img.shape

    xmin = best_box[0] * img_width
    ymin = best_box[1] * img_height
    box_width = best_box[2] * img_width
    box_height = best_box[3] * img_height
    pixel_box = [xmin,ymin,box_width,box_height]

    xmin_gt = gt_box[0] * img_width
    ymin_gt = gt_box[1] * img_height
    box_width_gt = gt_box[2] * img_width
    box_height_gt = gt_box[3] * img_height
    pixel_box_gt = [xmin_gt,ymin_gt,box_width_gt,box_height_gt] 

    return pixel_box, pixel_box_gt
