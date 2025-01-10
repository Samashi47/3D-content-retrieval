from logic.objloader import OBJ
from logic.descriptors.zernike import zernike_moments, compute_distance_zernike
from logic.descriptors.fourier import compute_descriptor, compute_distance_fourier
from pymongo import MongoClient
import numpy as np
import random
import shutil
import os
import dotenv


def create_query_db(source_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    else:
        shutil.rmtree(dest_folder)
        os.makedirs(dest_folder)

    for category in os.listdir(source_folder):
        category_path = os.path.join(source_folder, category)
        if not os.path.isdir(category_path):
            continue

        models = [
            model for model in os.listdir(category_path) if model.endswith(".obj")
        ]
        num_models = len(models)
        num_to_select = random.randint(1, min(10, num_models))
        random_models = random.sample(models, num_to_select)

        for model_name in random_models:
            src_path = os.path.join(category_path, model_name)
            dest_path = os.path.join(dest_folder, model_name)
            shutil.copy(src_path, dest_path)


def connect_to_db():
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client["3DPotteryDataset"]
    collection = db["descriptors"]
    return collection


def process_database_models(root_folder, not_indexed):

    collection = connect_to_db()

    for category in os.listdir(root_folder):
        category_path = os.path.join(root_folder, category)
        if not os.path.isdir(category_path):
            continue
        for model_name in os.listdir(category_path):
            model_path = os.path.join(category_path, model_name)
            model = OBJ(model_path, enable_opengl=False)
            if model is None:
                continue
            if model_name in not_indexed:
                continue
            if collection.find_one({"model_name": model_name.replace(".obj", "")}):
                continue
            print(f"Processing {model_name} from {category}...")
            zernike = zernike_moments(model, order=3, scale_input=True)
            fourier = compute_descriptor(model, N=128, K=2)

            # Store in MongoDB
            doc = {
                "model_name": model_name.replace(".obj", ""),
                "category": category,
                "zernike": zernike,
                "fourier": fourier.tolist(),
            }
            collection.insert_one(doc)


def compute_distances(query_desc, db_desc):
    """Calculate distances between query models and database modelss for each feature"""

    distances = []
    for doc in db_desc:
        dist_metrics = {
            "model_name": doc["model_name"],
            "category": doc["category"],
            "zernike_distance": compute_distance_zernike(
                query_desc["zernike"], np.array(doc["zernike"]), metric="l2"
            ),
            "fourier_distance": compute_distance_fourier(
                query_desc["fourier"], np.array(doc["fourier"]), metric="l2"
            ),
        }
        distances.append(dist_metrics)
    return distances


def min_max_normalize(values):
    """Normalize values to range [0,1] using min-max scaling"""
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(x - min_val) / (max_val - min_val) for x in values]


def calculate_similarities(distances, weights=None):
    """Calculate final similarity scores using normalized distances"""
    if weights is None:
        weights = {"zernike": 0.5, "fourier": 0.5}

    # Extract distances for each feature
    zernike_distances = [d["zernike_distance"] for d in distances]
    fourier_distances = [d["fourier_distance"] for d in distances]

    # Normalize distances
    norm_zernike = min_max_normalize(zernike_distances)
    norm_fourier = min_max_normalize(fourier_distances)

    for idx, dist_metrics in enumerate(distances):
        # Calculate weighted sum of normalized distances
        dist_metrics["similarity"] = 1 - (
            weights["zernike"] * norm_zernike[idx]
            + weights["fourier"] * norm_fourier[idx]
        )

    distances.sort(key=lambda x: x["similarity"], reverse=True)
    return distances


def get_top_images(similarities, x=5):
    top = similarities[:x]
    return [(doc["model_name"], doc["similarity"]) for doc in top]


def process_query_model(dest_folder, rand_img):
    query_path = os.path.join(dest_folder, rand_img)
    model = OBJ(query_path, enable_opengl=False)
    if model is None:
        return None
    zernike = zernike_moments(model, order=3, scale_input=True)
    fourier = compute_descriptor(model, N=128, K=2)
    return {"zernike": zernike, "fourier": fourier.tolist()}


def rename_files_to_lowercase(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, filename.lower())
            os.rename(old_path, new_path)


def RetrieveModels(folder, query_desc, rand_model, n=5):
    if query_desc is None:
        print("Invalid query image.")
    else:
        collection = connect_to_db()
        db_desc = list(collection.find())
        distances = compute_distances(query_desc, db_desc)
        similarities = calculate_similarities(distances)
        top_images = get_top_images(similarities, n)

        return top_images, distances
