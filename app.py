from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import torch
import numpy as np
from utils.build_model import DayNightClassifier, transform_image
import os

app = Flask(__name__)
api = Api(app)
# create new model object
model = DayNightClassifier()

# defining model weights path
weights_path = 'lib/models/'
weights = os.listdir(os.path.join(os.getcwd(), 'lib/models'))

if len(weights) == 0:
    raise OSError("Can't find model_weights to load")
else:
    weights = weights_path + str(weights[0])

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('img', type=str)
parser.add_argument('weights_path', type=str, default=weights)


class PredictScene(Resource):
    def get(self):        
        # use parser and find the user's query
        args = parser.parse_args()

        img = args['img']
        weights_path = args['weights_path']

        # load trained classifier
        model.load_state_dict(torch.load(weights_path))
    
        # Predict
        with torch.no_grad():
            model.eval()
            img_t = transform_image(img)
            pred_ = model.forward(img_t)
            pred_ = float(pred_.cpu().detach().numpy())

            if pred_ < 0.5:
                pred_img = 'Night'
                confidence = 1 - pred_

            else:
                pred_img = 'Day'
                confidence = pred_

        # create JSON object
        output = {'prediction: ': pred_img, 'confidence: ': confidence}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictScene, '/')


if __name__ == '__main__':
    app.run(debug=False)
