#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
Module documentation:

Project           : Electric and water meter detection
Program name      : train.py
Author            : Bjarne Meyn 
Date created      : 20191016
description       : Script to create a new Custom Vision Project, upload training images,
                    train and puplish the model
'''

# Imports
import os
import pandas as pd
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region

'''
ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"

# Replace with a valid key
subscription_key = "f27bc63d4a8944e687627072687a73c3"
prediction_resource_id = "/subscriptions/f7256e7b-4859-4bf1-af37-6d8bbf16129f/resourceGroups/MSBI-Hackathlon-2019-2ea64/providers/Microsoft.CognitiveServices/accounts/cvprediction"
publish_iteration_name = "detectModel"
project_name = "WaterMeterDetection"
'''

ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"


# Replace with a valid key
subscription_key = "08059cd22a544faf992751fd92344b88"
#prediction_resource_id = "/subscriptions/f7256e7b-4859-4bf1-af37-6d8bbf16129f/resourceGroups/MSBI-Hackathlon-2019-2ea64/providers/Microsoft.CognitiveServices/accounts/cvprediction"
#publish_iteration_name = "detectModel"
project_name = "Objekterkennung"
project_id = "2094ff6f-868e-45c2-8ac3-92cd3bbcebac"

trainer = CustomVisionTrainingClient(subscription_key, endpoint=ENDPOINT)

print(trainer.config.api_key)
print(trainer.config.endpoint)

# Find the object detection domain
#obj_detection_domain = next(domain for domain in trainer.get_domains() if domain.type == "ObjectDetection" and domain.name == "General")


# Create a new project
#print ("Creating project...")

project = trainer.get_project(project_id)
print(project.id)

# Create a new tag
tag_str_list = "NumberArea"
NumberArea_tag = trainer.create_tag(project.id, tag_str_list)


#! Upload Images
# Update this with the path to where you downloaded the images.
train_images_path = "./CustomVision/data/train_images/"
test_images_path = "./CustomVision/data/test_images/"
label_path = "./CustomVision/data/labels/image_regions.csv"
train_file_names = os.listdir(train_images_path)
labels_df = pd.read_csv(label_path)


# Go through the data table above and create the images
print("Adding images...")

#! Image need to uploaded in batches with max size of 64 samples
# function to create the batches
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# use the batch function to create batches
batch_num = 0
for batch_samples in batch(train_file_names, 64):
    batch_num =+ 1
    print("Starting Batch: " + str(batch_num) + " / " + str((len(train_file_names)/64 )+1))
    print(len(batch_samples))

    tagged_images_with_regions = []

    # iterate throw all batches
    for file_name in batch_samples:
        # select the region labels from the image_redions.csv by the image_id
        row = labels_df[labels_df["image_id"]==file_name]
        
        # save the labels from the dataframe into variables
        x,y,w,h =  row["left"],row["top"], row["width"],  row["height"]
        image_id = row["image_id"]

        # create the CustoVision Region objects 
        regions = [Region(tag_id=NumberArea_tag.id, left=x,top=y,width=w,height=h)]

        # load the images and combine the image data withe the
        with open(train_images_path + file_name, mode="rb") as image_contents:
            tagged_images_with_regions.append(ImageFileCreateEntry(name=image_id, contents=image_contents.read(), regions=regions))

    upload_result = trainer.create_images_from_files(project.id, images=tagged_images_with_regions)

    if not upload_result.is_batch_successful:
        print("Image batch upload failed.")
        for image in upload_result.images:
            print("Image status: ", image.status)

'''
import time

print("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print("Training status: " + iteration.status)
    time.sleep(1)

# The iteration is now trained. Publish it to the project endpoint
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print("Done!")
'''
