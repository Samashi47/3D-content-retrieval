from flask import Response, request, Flask
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
CORS(app)


@app.route("/", methods=["POST"])
def home():
    print(request.json)
    return "Hello, World!"


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json
    email = data["email"]
    dotenv.load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client["RSSCN7"]
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
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
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
    db = client["RSSCN7"]
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


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")