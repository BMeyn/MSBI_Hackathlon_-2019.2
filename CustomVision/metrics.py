#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
Module documentation:

Project           : Electric and water meter detection
Program name      : metrics.py
Author            : Bjarne Meyn 
Date created      : 20191016
description       : Collection of metric functions to evaluate the machine learning model

'''

def IoU(a, b):  
    
    # returns 0 if rectangles don't intersect

    # rectangle = [left, top, width, height]
    
    # width of the intersection area
    dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])

    # height of the intersection area
    dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    
    # calculate the intersection area
    AnB = dx*dy
    
    # calculate the combined area of both regtangles (AuB)
    vx = b[3] * b[2] 
    vy = a[3] * a[2]
    AuB = vx + vy
    
    if (dx>=0) and (dy>=0):
        return AnB / (AuB- AnB)
    else:
        return 0