# Day-Night Classifier 

The model is built in **pytorch** and deployed as a **REST API** using **Flask** 

## Dependencies 

```bash
pip install -r requirements.txt 
```

## Dataset Collection

Scraped 1200+ samples for model training, validation & testing using `selenium` which then filterd to 1200 samples 

Dataset is collected for **13** different countries from Europe, USA & Australia

## Inputs

Input data are resized, normalized accross the RGB channels & transformed to pytorch tensors:
```python
transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```

## Model Training

Model is finetuned on **resnet-101** with **Adam** optimizer and hyperparameters:
```python
batch_size = 32
n_epochs = 32
learning_rate = 0.0001
betas = (0.9, 0.999)
```

## Model Outputs

### Train-Val Loss

<img src="media/loss.png" alt="Train-Val Loss">


### Train-Val Accuracy

<img src="media/acc.png" alt="Train-Val Accuracy"> <br>


## File Structure

- app.py: Flask API application
- build_model.py: Utilities for building the inference model 
- scraping_data.ipynb: Jupyter notebook for scraping the data
- train.classifier.ipynb: Jupyter notebook for training the model

- lib/models: Should contain **saved_weights** for the trained model

Please download the model weights from this : 
<a href="https://drive.google.com/drive/folders/1dcfz1sulyhUuEo7Pbtfk0h6-fspvT5Zj?usp=sharing"> link</a>
also, you'll find the scraped dataset through this link <br>

### For model training on colab notebook <br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UYwYAH1jHbZKKZvW-jZug6-A4fI0Aywy)

## Testing the API

1. Run the Flask API locally for testing. Go to directory with `app.py`

```bash
python app.py
```

2. In a new terminal window, use **HTTPie** to make a GET request at the URL of the API or use `curl` command for the same purpose

```bash
curl -X GET http://127.0.0.1:5000/ -d img="img_path" 

-X option for "Specifing request command to use"
```

Example of `curl` request
```bash
curl -X GET http://127.0.0.1:5000/ -d img='/home/ayman/scraped_dataset/day/9ea6801b53.jpg'
```


3. Example of successful output.

```bash
Serving Flask app "app" (lazy loading)
Environment: production
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead.
Debug mode: on
Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
Restarting with inotify reloader
Debugger is active!
Debugger PIN: 167-245-170
127.0.0.1 - - [15/Feb/2021 20:35:31] "GET / HTTP/1.1" 200 -

{
    "prediction: ": "Day",
    "confidence: ": 0.9999970197677612
}
```

