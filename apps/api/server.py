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
from logic.search import RetrieveModels, process_query_model


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
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
    # filename = request.json["filename"]
    filename = "Abstractshape1.obj"
    app.logger.info(f"changed to {filename}")
    uploads = os.path.join(app.root_path, "assets")
    app.logger.info(f"uploads: {uploads}")
    try:
        app.logger.info(f"Sending {filename} from {uploads}")
        return send_file(
            os.path.join(uploads, filename),
            mimetype="model/obj",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return str(e)


@app.route("/search", methods=["POST"])
def search():
    if "model" not in request.files:
        return Response(
            json.dumps({"message": "No file part"}),
            mimetype="application/json",
            status=400,
        )

    model = request.files["model"]
    number_of_results = request.form["numberOfResults"]
    if model.filename == "":
        return Response(
            json.dumps({"message": "No selected file"}),
            mimetype="application/json",
            status=400,
        )

    source_folder = "assets/3DPottery"
    thumbnails_folder = "Thumbnails"
    dest_folder = os.path.join(app.root_path, "temp")
    model.save(os.path.join(dest_folder, model.filename))
    model.close()
    query_desc = process_query_model(dest_folder, model.filename)
    print(number_of_results)
    # models, similarities = RetrieveModels(thumbnails_folder, query_desc, model.filename, n=number_of_results)
    return Response(
        json.dumps({"message": "File uploaded successfully"}),
        mimetype="application/json",
        status=201,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
