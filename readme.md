# Multiple Disease Prediction System

This is a web application for predicting multiple diseases including heart disease, diabetes, breast cancer, blood pressure, kidney disease, and COVID-19.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required modules from the requirements.txt file that is provided into the repository.

## Features

- Predicts multiple diseases including heart disease, diabetes, breast cancer, blood pressure, kidney disease, and COVID-19.
- Provides risk assessment based on input data or uploaded medical images.
- Easy-to-use web interface.

## Dataset

The datasets used for training the models are included in this repository. However, you can also find suitable datasets from various sources online.


## Code Snippet

```python
from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from werkzeug.utils import secure_filename
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

heart_model = pickle.load(open('webapp/model/heart.pkl', 'rb'))
bp_model = pickle.load(open('webapp/model/bp.pkl', 'rb'))
diabetes_model = pickle.load(open('webapp/model/diabetes.pkl', 'rb'))
bc_model = pickle.load(open('webapp/model/bc.pkl', 'rb'))
kd_model = load_model('webapp/model/kd.h5')
covid_model = load_model('webapp/model/covid.h5')

# Routes and prediction functions are defined here

if __name__ == '__main__':
    app.run(debug=True)

```

## Credits

This project is developed by [Soumyadeep Dawn](https://github.com/soumya18122002), Sambita Majumdar and Srijoni Kumar as a part of University of Engineering and Management, Kolkata. Special thanks to [Harsh Singh](https://github.com/karmathecoder).

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[SOUMYADEEP DAWN](https://github.com/soumya18122002/Multiple_Disease_Prediction_Using_Flask)