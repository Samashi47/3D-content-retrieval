from descriptors.zernike import zernike_moments
from objloader import OBJ
import numpy as np

obj1 = OBJ("3D Models/Native American - Bottle/Ark_HM_130_HI.obj", enable_opengl=False)
obj2 = OBJ("3D Models/Native American - Bowl/Ark_HM_775_HI.obj", enable_opengl=False)
obj3 = OBJ("3D Models/All Models/3DMillenium_bottle02.obj", enable_opengl=False)
order = 3
scale_input = True
descriptors1 = np.array(zernike_moments(obj1.vertices, obj1.faces, order, scale_input))
print(descriptors1)
descriptors2 = np.array(zernike_moments(obj2.vertices, obj2.faces, order, scale_input))
print(descriptors2)
descriptors3 = np.array(zernike_moments(obj3.vertices, obj3.faces, order, scale_input))
print(descriptors3)
print("Similarity between 1 and 2:", np.linalg.norm(descriptors1 - descriptors2))
print("Similarity between 1 and 3:", np.linalg.norm(descriptors1 - descriptors3))
