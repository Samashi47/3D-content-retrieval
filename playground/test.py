from descriptors.zernike import zernike_moments, compute_distance_zernike
from descriptors.fourier import compute_descriptor, compute_distance_fourier
from search import create_query_db, process_database_models, process_query_model, RetrieveModels
from objloader import OBJ
import numpy as np
import os
import random

source_folder = "3D Models"
dest_folder = "query_models"
thumbnails_folder = "Thumbnails"

# create the query database
# create_query_db(source_folder, dest_folder)

# populate the database
# if __name__ == "__main__":
#     not_indexed = os.listdir(dest_folder)
#     process_database_models(source_folder, not_indexed)

# obj1 = OBJ("3D Models/Native American - Bottle/Ark_HM_130_HI.obj", enable_opengl=False)
# obj2 = OBJ("3D Models/Native American - Bowl/Ark_HM_775_HI.obj", enable_opengl=False)
# obj3 = OBJ("3D Models/Native American - Jar/Ark_HM_55_Hi.obj", enable_opengl=False)

# # Zernike Moments
# order = 3
# scale_input = True
# descriptors1 = np.array(zernike_moments(obj1, order, scale_input))
# descriptors2 = np.array(zernike_moments(obj2, order, scale_input))
# descriptors3 = np.array(zernike_moments(obj3, order, scale_input))
# print("Similarity between 1 and 2 using Zernike Moments:", compute_distance_zernike(descriptors1, descriptors2, metric='l2'))
# print("Similarity between 1 and 3 using Zernike Moments:", compute_distance_zernike(descriptors1, descriptors3, metric='l2'))

# # Fourier Shape Descriptor
# N = 128
# K = 2
# feature_vector1 = compute_descriptor(obj1, N, K)
# feature_vector2 = compute_descriptor(obj2, N, K)
# feature_vector3 = compute_descriptor(obj3, N, K)
# print("Similarity between 1 and 2 using Fourier Shape Descriptor:", compute_distance_fourier(feature_vector1, feature_vector2, metric='l2'))
# print("Similarity between 1 and 3 using Fourier Shape Descriptor:", compute_distance_fourier(feature_vector1, feature_vector3, metric='l2'))

# Search
query_imgs = os.listdir(dest_folder)
rand_model = random.choice(query_imgs)
query_desc = process_query_model(dest_folder, rand_model)
RetrieveModels(thumbnails_folder, query_desc, rand_model, n=15)
