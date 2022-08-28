from flask import Flask, jsonify, request
from flask import render_template
from base64 import b64encode
import io
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__, template_folder='./html/')
hair_medulla_dictionary = ({0: 'Human', 1: 'Non-human'})
hair_root_dictionary = ({0: 'Anagen/Catagen - submit for DNA analysis', 1: 'Telogen type one - not suitable for DNA analysis', 2: 'Telogen type two/three - further analysis required'})

def load_medulla_model():
    medulla_model_conv = torchvision.models.resnet18(pretrained=True)
    for param in medulla_model_conv.parameters(): 
        param.requires_grad = False

    #parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = medulla_model_conv.fc.in_features
    medulla_model_conv.fc = nn.Linear(num_ftrs, 2)

    #define path
    PATH = 'models/medulla_model.pt'

    #load
    medulla_model_conv.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    medulla_model_conv.eval()
    return medulla_model_conv

def load_root_model():
    root_model_conv = torchvision.models.resnet18(pretrained=True)
    for param in root_model_conv.parameters(): 
        param.requires_grad = False

    #parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = root_model_conv.fc.in_features
    root_model_conv.fc = nn.Linear(num_ftrs, 3)

    #define path
    PATH = 'models/hair_root_model.pt'

    #load
    root_model_conv.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    root_model_conv.eval()
    return root_model_conv

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction_root(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = root_model_conv.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat.item()

def get_prediction_medulla(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = medulla_model_conv.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat.item()


root_model_conv = load_root_model()
medulla_model_conv = load_medulla_model()

@app.route('/predict_medulla', methods=['POST'])
def predict_medulla():
     if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        enc_img = b64encode(img_bytes).decode("utf-8")
        result = get_prediction_medulla(image_bytes=img_bytes)
        if result == 1:
            return render_template('resp_medulla_nonhuman.html', category = hair_medulla_dictionary[result], image = enc_img)
        return render_template('resp_medulla_human.html', category = hair_medulla_dictionary[result], image = enc_img)
   
@app.route('/predict_root', methods=['POST'])
def predict_root():
     if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        enc_img = b64encode(img_bytes).decode("utf-8")
        result = get_prediction_root(image_bytes=img_bytes)
        return render_template('resp.html', category = hair_root_dictionary[result], image = enc_img)

@app.route('/', methods=['GET'])
def indexfile():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()

