from flask import Flask, render_template, redirect, url_for, request
from helpers import Car_age, Old_or_New, Car_Usage_level, State_level, City_level
import pickle
import json
import numpy as np
from lightgbm import LGBMRegressor

app = Flask(__name__)

price = None
states = ['Texas','New York','Colorado','Utah','Florida','Connecticut','Idaho','North Dakota'
        ,'California','New Jersey','Ohio','Virginia','Indiana','Arizona','Oregon','Kansas'
        ,'Nebraska','Massachusetts','Maryland','Georgia','Minnesota','Hawaii','Louisiana'
        ,'New Mexico','Illinois','Alabama','Pennsylvania','South Carolina','North Carolina'
        ,'Washington','Wisconsin','Oklahoma','Kentucky','Mississippi','Missouri','Maine'
        ,'Arkansas','Michigan','Nevada','Tennessee','Florida','New Hampshire','Delaware'
        ,'West Virginia','Arizona','Vermont','South Dakota','Iowa','Rhode Island','Georgia'
        ,'Ohio','Montana','District of Columbia','Alaska','Virginia','Wyoming','Maryland'
        ,'California','Georgia']

makes = ['Acura', 'Alfa', 'AM', 'Aston', 'Audi', 'Bentley', 'BMW', 'Buick', 'Cadillac',
 'Chevrolet', 'Chrysler', 'Dodge', 'Ferrari' ,'FIAT', 'Fisker' ,'Ford',
 'Freightliner', 'Genesis', 'Geo' ,'GMC' ,'Honda', 'HUMMER', 'Hyundai',
 'INFINITI' ,'Isuzu' ,'Jaguar', 'Jeep' ,'Kia' ,'Lamborghini', 'Land', 'Lexus',
 'Lincoln' ,'Lotus' ,'Maserati' ,'Maybach' ,'Mazda', 'McLaren', 'Mercedes-Benz',
 'Mercury' ,'MINI', 'Mitsubishi' ,'Nissan' ,'Oldsmobile', 'Plymouth', 'Pontiac',
 'Porsche' ,'Ram' ,'Rolls-Royce' ,'Saab', 'Saturn' ,'Scion' ,'smart', 'Subaru',
 'Suzuki', 'Tesla' ,'Toyota', 'Volkswagen' ,'Volvo']
models_file = open('models.txt', 'r')
models = sorted(models_file.read().split(', '))
models_file.close()
cities_file = open('cities.txt', 'r')
cities = sorted(cities_file.read().split(', '))
cities_file.close()

# Opening JSON file
f = open('feature-transformation-data.json')
maps_dict = json.load(f)
f.close()

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
    purchase_year = int(request.form.get('purchase-year'))
    mileage = int(request.form.get('mileage'))
    make = request.form.get('make-name')
    model = request.form.get('model-name')

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
    return render_template("sign_in.html")

@app.route('/register')
def register():
    return render_template("sign_up.html")