from flask import Flask,request,jsonify,render_template
import pickle
from pyforest import *

app=Flask(__name__)

model = pickle.load(open('model_rf.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == "POST":
        
        features = [np.array([int(x) for x in request.form.values()])]
        prediction = model.predict(features)

        output = round(prediction[0],2)

        return render_template('index.html',prediction_text = 'Your Rating is : {}'.format(output))
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)