# Detection of brain boundaries
from Environment import *

def remove_edge(image_path, label_array, critical_value=112, interval=15):
	image = sitk.ReadImage(image_path)
	image_array = sitk.GetArrayFromImage(image)
	image_array[image_array > critical_value] = 255
	image_array[image_array <= critical_value] = 0
	binary_image = sitk.GetImageFromArray(image_array)
	binary_image = sitk.Cast(binary_image, sitk.sitkFloat32)
	edges = sitk.CannyEdgeDetection(binary_image, lowerThreshold=0.0, upperThreshold=40.0, variance = (5.0,5.0,5.0))
	edge_indexes = np.where(sitk.GetArrayViewFromImage(edges) == 1.0)
	physical_points = [edges.TransformIndexToPhysicalPoint([int(x), int(y), int(z)]) \
					for z,y,x in zip(edge_indexes[0], edge_indexes[1], edge_indexes[2])]
	edge_array = np.zeros_like(image_array)
	for i in range(len(physical_points)):
		edge_z, edge_y, edge_x = int(physical_points[i][2]), int(physical_points[i][1]), int(physical_points[i][0])
		edge_array[edge_z, edge_y-interval:edge_y+interval, edge_x-interval:edge_x+interval] = 300

	remove_edge_array = label_array + edge_array
	remove_edge_array[remove_edge_array==555]=0
	remove_edge_array[remove_edge_array==300]=0
	print('Complete removal of boundary')

	return remove_edge_array