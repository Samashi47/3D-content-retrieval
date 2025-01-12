from flask import Response, request, Flask, send_file, send_from_directory
from flask_cors import CORS
import json
import datetime
import hashlib
from pymongo.mongo_client import MongoClient
from bson import ObjectId
import jwt
import os
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.backends import default_backend
import base64
import dotenv
import logging
from logic.search import RetrieveModels, process_query_model, connect_to_db


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = "data:image/jpeg;base64," + base64.b64encode(
            image_file.read()
        ).decode("utf-8")
    return encoded_string


app = Flask(__name__)
app.logger.setLevel(logging.CRITICAL)
handler = logging.FileHandler("app.log")
app.logger.addHandler(handler)
CORS(app)


@app.route("/", methods=["POST"])
def home():
    app.logger.info("This is an INFO message")
    app.logger.debug("This is a DEBUG message")
    app.logger.warning("This is a WARNING message")
    app.logger.error("This is an ERROR message")
    app.logger.critical("This is a CRITICAL message")
    return "Hello, World!"


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json
    email = data["email"]
    dotenv.load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client["3DPotteryDataset"]
    collection = db["accounts"]
    user = collection.find_one({"email": email})
    if not user:
        return Response(
            json.dumps({"message": "User does not exist"}),
            mimetype="application/json",
            status=404,
        )

    hash_func = hashlib.sha256()
    hash_func.update(data["password"].encode())
    hashed_password = hash_func.hexdigest()

    if data["email"] == user["email"] and hashed_password == user["password"]:
        private_key = open(".ssh/jwt-key", "rb").read()
        key = load_pem_private_key(
            private_key, password=None, backend=default_backend()
        )
        payload = {
            "email": "test@test.com",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1),
        }
        token = jwt.encode(payload, key, algorithm="RS256")

        return Response(
            json.dumps({"token": token}), mimetype="application/json", status=200
        )
    else:
        return Response(
            json.dumps({"message": "Login Failed"}),
            mimetype="application/json",
            status=401,
        )


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.json
    email = data["email"]
    dotenv.load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client["3DPotteryDataset"]
    collection = db["accounts"]
    user = collection.find_one({"email": email})
    if user:
        return Response(
            json.dumps({"message": "User already exists"}),
            mimetype="application/json",
            status=409,
        )

    hash_func = hashlib.sha256()
    hash_func.update(data["password"].encode())
    hashed_password = hash_func.hexdigest()
    user = {
        "email": email,
        "password": hashed_password,
    }
    collection.insert_one(user)
    return Response(
        json.dumps({"message": "User created successfully"}),
        mimetype="application/json",
        status=201,
    )


@app.route("/download-model", methods=["GET", "POST"])
def download():
    filename = request.json["filename"] + ".obj"
    category = request.json["category"]
    uploads = os.path.join(app.root_path, "assets/3D Models")
    try:
        app.logger.info(f"Sending {filename} from {uploads}")
        return send_file(
            os.path.join(uploads, category, filename),
            mimetype="model/obj",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return str(e)


@app.route("/query-descriptors", methods=["POST"])
def query_descriptors():
    if "model" not in request.files:
        return Response(
            json.dumps({"message": "No file part"}),
            mimetype="application/json",
            status=400,
        )

    model = request.files["model"]
    if model.filename == "":
        return Response(
            json.dumps({"message": "No selected file"}),
            mimetype="application/json",
            status=400,
        )

    query_desc = process_query_model(model.stream)

    response = Response(json.dumps(query_desc), mimetype="application/json", status=200)

    return response


@app.route("/result-descriptors", methods=["POST"])
def result_descriptors():
    filename = request.json["model_name"]
    category = request.json["category"]
    collection = connect_to_db()
    model = collection.find_one({"model_name": filename, "category": category})
    response_data = {"zernike": model["zernike"], "fourier": model["fourier"]}

    response = Response(
        json.dumps(response_data), mimetype="application/json", status=200
    )

    return response


@app.route("/search", methods=["POST"])
def search():
    if "model" not in request.files:
        return Response(
            json.dumps({"message": "No file part"}),
            mimetype="application/json",
            status=400,
        )

    model = request.files["model"]
    number_of_results = int(request.form["numberOfResults"])
    if model.filename == "":
        return Response(
            json.dumps({"message": "No selected file"}),
            mimetype="application/json",
            status=400,
        )
    thumbnails_folder = "assets/Thumbnails"
    query_desc = process_query_model(model.stream)
    models = RetrieveModels(query_desc, n=number_of_results)
    results = [
        {
            "model_name": model["model_name"],
            "category": model["category"],
            "thumbnail": encode_image_to_base64(
                os.path.join(
                    thumbnails_folder, model["category"], model["model_name"] + ".jpg"
                )
            ),
            "similarity": model["similarity"],
        }
        for model in models
    ]
    response = Response(json.dumps(results), mimetype="application/json", status=200)

    return response


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
