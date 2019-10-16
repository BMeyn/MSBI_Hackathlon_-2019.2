from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os

# custom modules
from metrics import IoU
from visualization import plot_image_detection
from helperfunctions import calculate_pixel_box

credentials = {
    "endpoint": "https://eastus.api.cognitive.microsoft.com/",
    "prediction_key": "d4b69c14e9714e549dc360b045ccb54e",
    "prediction_resource_id": "https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/d6240099-dfb3-435c-ae92-84aba5149b12/detect/iterations/detectModel/image",
    "project_id": "d6240099-dfb3-435c-ae92-84aba5149b12",
    "publish_iteration_name": "detectModel"
}


def evaluation(
    image_name,
    credentials,
    test_images_path="./CustomVision/data/test_images/",
    label_path="./CustomVision/data/labels/image_regions.csv",
    vis=True):


    # save the credentials into variables
    ENDPOINT = credentials["endpoint"]
    prediction_key = credentials["prediction_key"]
    prediction_resource_id = credentials["prediction_resource_id"]
    project_id = credentials["project_id"]
    publish_iteration_name = credentials["publish_iteration_name"]


    # Now there is a trained endpoint that can be used to make a prediction
    predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

    # Open the sample image and get back the prediction results.
    with open(test_images_path + image_name, mode="rb") as test_data:
        results = predictor.detect_image(project_id, publish_iteration_name, test_data)

    # collect all probability scores.   
    select_best_score = [score.probability for score in results.predictions] 
    # select the highest score
    best_prediction = results.predictions[select_best_score.index(max(select_best_score))]
    # save the box regions of the highest score
    best_box = [
        float(best_prediction.bounding_box.left),
        float(best_prediction.bounding_box.top), 
        float(best_prediction.bounding_box.width),
        float(best_prediction.bounding_box.height)]

    # load the label data
    labels = pd.read_csv(label_path)
    # select the labels by the image name
    gt = labels[labels["image_id"] == image_name]
    # save the box regions of the label data
    gt_box = [
        float(gt.left),
        float(gt.top),
        float(gt.width),
        float(gt.height)]
    

    pixel_box, pixel_box_gt = calculate_pixel_box(best_box,gt_box, test_images_path, image_name)
    aoi_score = IoU(pixel_box, pixel_box_gt)

    if vis == True:
        plot_image_detection(image_name, test_images_path, best_box, gt_box)

    prediction_result = {
        "image_name": image_name,
        "image_path": test_images_path, 
        "probability":best_prediction.probability,
        "relative_bounding_box": {"left": best_box[0], "top": best_box[1], "box_width": best_box[2], "box_height": best_box[3]},
        "pixel_bounding_box": {"left": pixel_box[0], "top": pixel_box[1], "box_width": pixel_box[2], "box_height": pixel_box[3]},
        "aoi_score": aoi_score
    }

    return prediction_result


for img_name in os.listdir(r"CustomVision\data\test_images"):
    
    prediction = evaluation(image_name=img_name, credentials=credentials, vis=False)
    print(prediction)