from flask import Flask, request, send_file, jsonify
from bson import json_util
import base64
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from emotions import display


global expression

app = Flask(__name__)
CORS(app)

def parse_json(data):
    return json.loads(json_util.dumps(data))

# get all feilds
@app.route('/song', methods=('GET', 'POST'))
def allDt():
    global expression
    print( request.files)
    print("-------->")
    print(expression)
    base64_audio = ''

    with open('./honor-and-sword-main-11222.mp3', 'rb') as file:
        audio_data = file.read()
        base64_audio = base64.b64encode(audio_data).decode('utf-8')

    print(base64_audio)
    return jsonify(base64_audio)
    # Set the filename and content type
    # filename = open('./honor-and-sword-main-11222.mp3')
    # mimetype = 'audio/mp3'

    # Return the file as a response
    # return send_file(filename, as_attachment=True, mimetype=mimetype)
    # dd = request.form.get('image')
    # if 'image' not in request.files:
    #     return 'No file uploaded', 400
    
    # file = request.files['image']
    # print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    # print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    # file.save('sss.png')
    # return 'File uploaded successfully', 200

@app.route('/image', methods=['POST'])
def process_image():

    global expression
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    base64_image = request.form.get('image')
    imgdata = base64.b64decode(base64_image.split(',')[1])
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    # opencv One 
    nparr = np.frombuffer(imgdata, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("open.png", img)

    # matplot One
    # imgdata = base64.b64decode(base64_image.split(',')[1])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(plt.imread(BytesIO(imgdata)))
    # fig.savefig("mat.png")
    expression = display()
    print(display())

    return jsonify(expression)
