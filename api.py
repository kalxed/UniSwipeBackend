from flask import Flask, request, jsonify
from flask_cors import CORS
from model import * # TODO import necessary functions

app = Flask(__name__)
CORS(app)

@app.route('/compute', methods = ['POST'])
def compute(): 
    data = request.json 

    # accept inputs
    swipe_direction = data.get('swipe_direction')

    # set result
    result = testfcn([swipe_direction])
    json_result = jsonify({'result': result})

    return json_result

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port=5000, debug = True)