# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 19:50:56 2020

@author: Adithya

Perform augmentation on 3D point cloud
Augmentation done using Rotation and Scaling
pypcd is used to extract 3d point cloud data into numpy arrays

"""

import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from pypcd import pypcd
from scipy.spatial.transform import Rotation as R
from mpl_toolkits import mplot3d
import random
#tf.random.set_seed(1234) 

def perform_pcl_augmentation():
	#source dataset
	DATA_DIR = "./restaurant_object_dataset" #/Mug/Mug_Object11.pcd"
	#Target Dataset
	save_dir = "./Augmented_Dataset" 
	categories = os.listdir(DATA_DIR)
	for obj_class in categories:
		file_name_count = 0
		file_path = os.path.join(DATA_DIR,obj_class)
		all_files = os.listdir(file_path)
		file_count = len(os.listdir(os.path.join(save_dir,obj_class)))
		
		#Add files till folder has 100 files atleast
		while(file_count < 100):
			for file in all_files:
				if file[-3:] == 'pcd':
					pcd_file = os.path.join(file_path,file)
					cloud = pypcd.PointCloud.from_path(pcd_file)
					
					#Extract point cloud using pypcd lib
					new_cloud_data = cloud.pc_data.copy()
					new_cloud_data = cloud.pc_data.view(np.float32).reshape(cloud.pc_data.shape + (-1,))
					
					#Plot 3D point cloud
					fig = plt.figure()
					ax = plt.axes(projection='3d')
					ax.scatter3D(new_cloud_data[:,0], new_cloud_data[:,1], new_cloud_data[:,2])
					plt.show()
					
					for i in range(1,4):
						new_cloud_data = cloud.pc_data.copy()
						new_cloud_data = cloud.pc_data.view(np.float32).reshape(cloud.pc_data.shape + (-1,))
						
						new_cloud_data = np.delete(new_cloud_data, [3], axis=1)
						
					
						#Add random rotation along x y and z axes
						rotation_axis = ['x','y','z']
						for axis in rotation_axis:
							rotation_angle = random.randint(10,360)
							r = R.from_euler(axis, rotation_angle, degrees = True)
							new_cloud_data = r.apply(new_cloud_data)	
						#new_cloud_data = np.matmul(new_cloud_data,transformation_matrix)
						
						#Perform scaling along 3 different axes
						transformation_matrix = np.zeros([4,4])
						scale_matrix = [random.uniform(0.8,1.2), random.uniform(0.8,1.2), random.uniform(0.8,1.2)]
						new_cloud_data[:0] = new_cloud_data[:0] * scale_matrix[0]
						new_cloud_data[:1] = new_cloud_data[:1] * scale_matrix[1]
						new_cloud_data[:2] = new_cloud_data[:2] * scale_matrix[2] 
						
						
						fig = plt.figure()
						
						#Plot sclaed figures
						ax = plt.axes(projection='3d')
						ax.scatter3D(new_cloud_data[:,0], new_cloud_data[:,1], new_cloud_data[:,2],c = cloud.pc_data['rgba'])
						plt.show()
					
					new_cloud_data = new_cloud_data.astype('float32')
					
					#Add values to object.
					
					rgba = pypcd.decode_rgb_from_pcl(cloud.pc_data["rgba"])
					encoded_colors = pypcd.encode_rgb_for_pcl((rgba * 255).astype(np.uint8))
					new_data = np.hstack((new_cloud_data[:,0].reshape(1,-1).T, new_cloud_data[:,1].reshape(1,-1).T, new_cloud_data[:,2].reshape(1,-1).T, encoded_colors[:, np.newaxis]))
					
					#Create .pcd file from data
					new_cloud = pypcd.make_xyz_rgb_point_cloud(new_data)
					file_count = len(os.listdir(os.path.join(save_dir,obj_class)))
					file_count = file_count + 1 
					
					#Save .pcd file to folder
					file_save_path = os.path.join(save_dir, obj_class, obj_class + '_Object'+str(file_count) +'.pcd')
					new_cloud.save(file_save_path)
	
if __name__ == '__main__':
	perform_pcl_augmentation()
