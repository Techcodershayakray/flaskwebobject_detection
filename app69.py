from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import wikipediaapi
from bardapi import Bard
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def remove_duplicates(input_array):
    return list(set(input_array))
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
name=[]
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
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
               
                confidence = confidences[i]
                
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
        cv2.imwrite(result_image_path, img)
        
        for i in range(len(boxes)):
            if i in indexes:
                class_id = class_ids[i]
                class_name = classes[class_id]
                name=classes[class_id]
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
    
        n=remove_duplicates(name)
        topics=[]
        str1=""
    for i in range(len(n)-1):
        str1=str1+n[i]+","
        str1="relationship between "+str1+ "and "+n[len(n)-1]+"(only theory no venn diagram or any diagram)"
    for i in range(len(n)):
       topics.append(n[i])
       topics.append(str1)
       os.environ['_BARD_API_KEY']="Zgg0wtyIOTSwSXpQ4l1nnXcyCGlZZGY8AnviguSOHbxI74vvkvi0zaIaKOfKoJOVGE7CMA."
       output=[]
       array=[]
    for i in topics:
       bard_output = Bard().get_answer(i)['content']
       array.append(i)
       output.append(bard_output)
       output.append(" ")
       print(n)
          
    return render_template('uploaded.html', filename='result_' + file.filename, detected_objects=detected_objects,array=array)


if __name__ == '__main__':
    app.run(debug=True)
