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
if __name__ == "__main__":
    query_imgs = os.listdir(dest_folder)
    to_test = [
        "00110054.obj",
        "3dmillenium_bottle01.obj",
        "3dmillenium_bowl01.obj",
        "3dmillenium_bowl03.obj",
        "5dsom_fakej.obj",
        "abstractshape35.obj",
        "abstractshape5.obj",
        "alabastron24.obj",
        "alabastron48.obj",
        "amphora14.obj",
        "amphora16.obj",
        "amphora501ncon.obj",
        "ark_hm_107_hi.obj",
        "ark_hm_1300_hi.obj",
        "ark_hm_202_hi.obj",
        "ark_hm_226_hi.obj",
        "ark_hm_309_hi.obj",
        "ark_hm_33_hi.obj",
        "ark_hm_453_hi.obj",
        "ark_hm_518_hi.obj",
        "likithosqp.obj",
        "london b 658.obj",
        "london d 20.obj",
        "london d 51.obj",
        "london d 58.obj",
        "london d 61.obj",
        "london e 180.obj",
        "london e 233.obj",
        "london f 90_1.obj",
        "london f 90_5.obj",
        "psykter0.obj",
        "psykter17.obj",
        "psykter2.obj",
        "psykter27.obj",
        "psykter34.obj",
        "psykter6.obj",
        "tsi_enst_fr_cp_vase_1305b_mr.obj",
        "tsi_enst_fr_cp_vase_1305_lr.obj",
        "visiting_grave.obj",
    ]
    for i, img in enumerate(query_imgs, start=1):
        if img not in to_test:
            continue
        print(f"------------------- {i} -------------------")
        print("Query image: ", img)
        query_desc = process_query_model(dest_folder, img)
        if query_desc["zernike"] == []:
            query_desc["zernike"] = np.zeros(6)
        top_images, _ = RetrieveModels(thumbnails_folder, query_desc, img, n=20)
        print("Top images: ", top_images)
