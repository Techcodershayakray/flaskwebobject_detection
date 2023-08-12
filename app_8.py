from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import wikipediaapi
from bs4 import BeautifulSoup
import requests
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv3
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
# Wikipedia API setup
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="app/1.0"  # Replace with your app's name and version
)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Object Detection
        img = cv2.imread(filename)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        detected_objects = []

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
                if i in indexes:
                    class_id = class_ids[i]
                    class_name = classes[class_id]
                    wiki_page = wiki_wiki.page(class_name)
                
                    detected_object = {
                    'class': class_name,
                    'confidence': confidences[i],
                    'box': boxes[i],
                    'summary': wiki_page.summary if wiki_page.exists() else "No Wikipedia data available"
                }
                    detected_objects.append(detected_object)
                filename = filename.split('_')[-1]
                print(filename)  
        return render_template('uploaded.html', filename='result_' + file.filename, detected_objects=detected_objects)


if __name__ == '__main__':
    app.run(debug=True)
