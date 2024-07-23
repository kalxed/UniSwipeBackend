from flask import Flask, request, jsonify
from flask_cors import CORS
from model import SchoolPicker
from model import DQNStudent
from model import process_swipe

app = Flask(__name__)
CORS(app)

env = SchoolPicker(drop_columns=True)
agent = DQNStudent(state_size=env.num_features, action_size=len(env.remaining))

state = env.restart()

@app.route("/recommend", methods=["GET"])
def recommend_college():
    college = env.information_on_current_school()
    response = {
        "schoolName": str(env.current_school),
        "schoolCity": str(college["CITY"]),
        "schoolState": str(college["STABBR"]),
        "schoolZip": str(college["ZIP"]),
        "schoolTuition": str(college["TUITIONFEE_IN"]),
        "schoolUndergradPopulation": str(college["UGDS"]),
        "schoolWebsite": str(college["INSTURL"]),
        "schoolAcceptanceRate": str(round(100 * college["ADM_RATE"], 2)) + "%",
        "schoolTopMajor": "N/A",
    }
    return jsonify(response)


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    swipe_direction = data.get("swipe_direction")
    process_swipe(swipe_direction, agent, state, env)
    college = env.information_on_current_school()
    response = {
        "schoolName": str(env.current_school),
        "schoolCity": str(college["CITY"]),
        "schoolState": str(college["STABBR"]),
        "schoolZip": str(college["ZIP"]),
        "schoolTuition": str(college["TUITIONFEE_IN"]),
        "schoolUndergradPopulation": str(college["UGDS"]),
        "schoolWebsite": str(college["INSTURL"]),
        "schoolAcceptanceRate": str(round((100 * college["ADM_RATE"]), 2)) + "%",
        "schoolTopMajor": "N/A",
    }
    print(response)
    return jsonify(response)


if __name__ == "__main__":
    app.run()
