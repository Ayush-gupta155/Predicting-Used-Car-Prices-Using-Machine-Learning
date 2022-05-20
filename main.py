from flask import Flask, render_template, redirect, url_for, request, jsonify
import pickle
import json
import numpy as np
from lightgbm import LGBMRegressor
from transformation_functions import *



app = Flask(__name__)

price = None

# Opening JSON files
f = open('feature-transformation-data.json')
maps_dict = json.load(f)
f.close()

f = open('models_to_makes.json')
models_to_makes = json.load(f)
f.close()

f = open('cities_to_states.json')
cities_to_states = json.load(f)
f.close()

states = sorted(list(cities_to_states.keys()))
makes = sorted(list(models_to_makes.keys()))
cities = sorted(list(cities_to_states[states[0]]))
models = sorted(list(models_to_makes[makes[0]]))

# load the model from disk
ml_model = pickle.load(open('lgb', 'rb'))

def transform_features(city, state, year, mileage, make, model):
    features = {}
    encoded_city = maps_dict['city-value-count'][city]
    encoded_state = maps_dict['state-value-count'][state]
    features['MFG_YEAR'] = year
    features['CAR_AGE'] = Car_age(year)
    features['MILEAGE'] = mileage
    features['CAR_DRIVEN_PER_YEAR'] = mileage / features['CAR_AGE']
    features['STATE'] = state    
    features['MAKE'] = make
    features['MODEL'] = model
    features['CITY'] = city
    features['STATE_IMPORTANCE'] = State_level(encoded_state)
    features['CITY_IMPORTANCE'] = City_level(encoded_city)
    features['OLD_OR_NEW'] = Old_or_New(features['CAR_AGE'])
    features['CAR_USAGE_LEVEL'] = Car_Usage_level(features['CAR_DRIVEN_PER_YEAR'])

    for column in maps_dict['categorical-cols']:
        features[f'{column}_ENCODED'] = maps_dict[f"{column}-avg-price"][features[column]]
        features[f'{column}_ENCODED'] = np.log(features[f'{column}_ENCODED'])

    data_sample = []
    for column in maps_dict['final-data-columns']:
        if (column != 'PRICE') and (column != 'ID'):
            data_sample.append(features[column])

    return data_sample

@app.route('/')
def index():
    title = "Know your Car"
    return render_template('index.html', titel=title, price=price, states=states,
                            cities=cities, makes=makes, models=models)

@app.route('/getprice', methods = ['POST'])
def getPrice():
    city = request.form.get('city-name')
    state = request.form.get('state-name')
    make = request.form.get('make-name')
    model = request.form.get('model-name')
    try:
        purchase_year = int(request.form.get('purchase-year'))
        mileage = int(request.form.get('mileage'))
    except ValueError:
        redirect(url_for('index'))

    sample = transform_features(city=city, state=state, year=purchase_year, mileage=mileage, make=make, model=model)
    print(sample)
    global price
    price = np.round(ml_model.predict(np.array(sample).reshape(1, -1))[0], 2)
    return redirect(url_for('index'))

@app.route('/about-us')
def about():
    return render_template("about.html")

@app.route('/contact-us')
def contact():
    return render_template("contact_us.html")

@app.route('/login')
def login():
    return render_template("Sign_in.html")

@app.route('/register')
def register():
    return render_template("sign_up.html")

@app.route('/city/<state>')
def city(state):
    return jsonify({'cities': sorted(list(cities_to_states[state]))})

@app.route('/model/<make>')
def model(make):
    return jsonify({'models': sorted(list(models_to_makes[make]))})