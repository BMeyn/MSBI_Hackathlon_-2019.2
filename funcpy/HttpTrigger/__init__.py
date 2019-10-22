import logging
import requests
import time
from PIL import Image
from io import BytesIO
import azure.functions as func
import json

def ComputerVisionPredict(image_path):

    subscription_key = "076cf83a447c4411a125dc952562e788"
    endpoint = "https://eastus.api.cognitive.microsoft.com/"

    text_recognition_url = endpoint + "vision/v2.0/RecognizeText"

    # Set image_url to the URL of an image that you want to analyze.
    image_url = image_path

    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    data = {'url': image_url}
    params = {'mode' : 'Handwritten'}

    response = requests.post(text_recognition_url, headers=headers, json=data, params=params)
    response.raise_for_status()

    # Extracting text requires two API calls: One call to submit the
    # image for processing, the other to retrieve the text found in the image.

    # Holds the URI used to retrieve the recognized text.
    operation_url = response.headers["Operation-Location"]

    # The recognized text isn't immediately available, so poll to wait for completion.
    analysis = {}
    poll = True
    while (poll):
        response_final = requests.get(response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        time.sleep(1)
        if ("recognitionResult" in analysis):
            poll = False
        if ("status" in analysis and analysis['status'] == 'Failed'):
            poll = False

    polygons = []
    if ("recognitionResult" in analysis):
        lines = analysis["recognitionResult"]["lines"]
        # Extract the recognized text, with bounding boxes.
        polygons = [(line["boundingBox"], line["text"]) for line in analysis["recognitionResult"]["lines"]]

    return analysis


def find_matching_box(box1, box2):

    # box locations:
    # computer vision : X: top left, Y top left, X: top right, Y: top right, X bunn 

    return




def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')

    # get the prediction of the ObjectDetection
    ObjetDetectionPred = req.params.get('ObjetDetectionPred')
    
    
    # if name is not provided as params search in the body
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')
    
    # if ObjetDetectionPred is not provided as params search in the body
    if not ObjetDetectionPred:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            ObjetDetectionPred = req_body.get('ObjetDetectionPred')
    
    # select the best Object Detection prediction 

    if ObjetDetectionPred:
        seq = [x['probability'] for x in ObjetDetectionPred]
        best_cv_score = max(seq)
        best_cv_prediction = ObjetDetectionPred[seq.index(best_cv_score)]


    # combine the BlobStorage path with the image name that has been upload to the storage
    storage_path = "https://storageaccmsbi.blob.core.windows.net"

    image_path = storage_path + name

    result = ComputerVisionPredict(image_path)
    result_json = json.dumps(result)
    if name:
        return func.HttpResponse(result_json)
    else:
        return func.HttpResponse(
             "Please pass a name on the query string or in the request body",
             status_code=400
        )


