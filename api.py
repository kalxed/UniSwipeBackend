from flask import Flask, request, jsonify
from model import recommender
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/recommend", methods=["GET"])
def recommend_college():
    college = recommender.recommend()
    recommender.last_recommended = college
    response = {
        "schoolName": str(college["INSTNM"]),
        "schoolCity": str(college["CITY"]),
        "schoolState": str(college["STABBR"]),
        "schoolZip": str(college["ZIP"]),
        "schoolTuition": str(college["TUITIONFEE_IN"]),
        "schoolWebsite": str(college["INSTURL"]),
        "schoolAcceptanceRate": str(college["ADM_RATE"]),
        "schoolTopMajor": "N/A",
    }
    return jsonify(response)


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    swipe_direction = data.get("swipe_direction")
    college = recommender.handle_feedback(swipe_direction)
    response = {
        "schoolName": str(college["INSTNM"]),
        "schoolCity": str(college["CITY"]),
        "schoolState": str(college["STABBR"]),
        "schoolZip": str(college["ZIP"]),
        "schoolTuition": str(college["TUITIONFEE_IN"]),
        "schoolWebsite": str(college["INSTURL"]),
        "schoolAcceptanceRate": str(college["ADM_RATE"]),
        "schoolTopMajor": "N/A",
    }
    print(response)
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=5000,debug=True)
