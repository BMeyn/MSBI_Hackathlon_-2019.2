from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
import pandas as pd


ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"
prediction_key = "d4b69c14e9714e549dc360b045ccb54e"
prediction_resource_id = "https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/d6240099-dfb3-435c-ae92-84aba5149b12/detect/iterations/detectModel/image"

project_id = "d6240099-dfb3-435c-ae92-84aba5149b12"
publish_iteration_name = "detectModel"
test_images_path = "./CustomVision/data/test_images/"
label_path = "./CustomVision/data/labels/image_regions.csv"


# Now there is a trained endpoint that can be used to make a prediction
predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

# Open the sample image and get back the prediction results.
with open(test_images_path + "Image_91.jpg", mode="rb") as test_data:
    results = predictor.detect_image(project_id, publish_iteration_name, test_data)

# Display the results.   
select_best_score = [] 
for prediction in results.predictions:
    #print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
    select_best_score.append(prediction.probability)
best_prediction = results.predictions[select_best_score.index(max(select_best_score))]

best_box = [
    float(best_prediction.bounding_box.left),
    float(best_prediction.bounding_box.top), 
    float(best_prediction.bounding_box.width),
    float(best_prediction.bounding_box.height)]

print(best_box)

labels = pd.read_csv(label_path)

gt = labels[labels["image_id"] == "Image_91.jpg"]

gt_box = [
    float(gt.left),
    float(gt.top),
    float(gt.width),
    float(gt.height)]

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

img = cv2.imread(test_images_path + "Image_91.jpg")


fig, ax = plt.subplots(1)
ax.imshow(img)


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

rect = patches.Rectangle((xmin, ymin), box_width,box_height, linewidth=2, edgecolor="r", fill=False)
rect_gt = patches.Rectangle((xmin_gt, ymin_gt), box_width_gt,box_height_gt, linewidth=2, edgecolor="g", fill=False)

ax.add_patch(rect)
ax.add_patch(rect_gt)
plt.show()

def AOI(a, b):  # returns None if rectangles don't intersect
    
    # width of the intersection area
    dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    # height of the intersection area
    dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    # calculate the intersection area
    AnB = dx*dy
    print(AnB)

    # calculate the combined area of both regtangles
    vx = b[3] * b[2] 
    vy = a[3] * a[2]
    AuB = vx + vy
    print(AuB)
    if (dx>=0) and (dy>=0):
        return AnB / AuB
aoi_result = AOI(pixel_box, pixel_box_gt)
print(aoi_result)