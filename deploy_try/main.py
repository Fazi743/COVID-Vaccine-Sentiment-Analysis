from flask import Flask, request, jsonify
import pickle
import numpy as np
import sklearn
Model = pickle.load(open('RandomForest_1.pkl', 'rb'))
app = Flask(__name__)
#
@app.route('/')
def home():
    return "Hello"

@app.route('/predict', methods=['POST'])
def predict():
    sex = (int)(request.form.get('sex'))
    wat = (int)(request.form.get('Height'))
    hit = (int)(request.form.get('Weight'))
    umer = (int)(request.form.get('age'))
    dibt = (int)(request.form.get('diabetes'))
    goal = (int)(request.form.get('Goal'))


    input_query = np.array([[sex, wat, hit, umer, dibt, goal]])

    result = Model.predict(input_query)

    return result.tolist()

if __name__ == '__main__':
    # http: //172.0.6.50: 5000
    # 192.168.43.34
    #app.run(debug=True)
    app.run(host = "0.0.0.0")



