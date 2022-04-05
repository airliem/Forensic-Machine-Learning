from flask import Flask, jsonify, request
from flask import render_template
import io
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)
hair_dictionary = ({0: 'Anagen/Catagen - submit for DNA analysis', 1: 'Telogen type one - not suitable for DNA analysis', 2: 'Telogen type two/three - further analysis required'})

#model_conv = torchvision.models.resnet18(pretrained=True)
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters(): 
    param.requires_grad = False

#parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 3)

#define path
PATH = 'state_dict_model.pt'

#load
model_conv.load_state_dict(torch.load(PATH))
model_conv.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model_conv.forward(tensor)
    _, y_hat = outputs.max(1)
    return hair_dictionary[y_hat.item()]

@app.route('/predict', methods=['POST'])
def predict():
     if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        return get_prediction(image_bytes=img_bytes)
        #return jsonify({'class_name': get_prediction(image_bytes=img_bytes)})

@app.route('/', methods=['GET'])
def indexfile():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()

