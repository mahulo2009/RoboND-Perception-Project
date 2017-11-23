#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from std_srvs.srv import Empty
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(ros_msg):

    ##TODO REMOVE PREVIOUS MAPS  
    rospy.wait_for_service('/clear_octomap')
    try:
        clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
	clear_octomap()
        print ("Response: OK" )     
    except rospy.ServiceException, e:
        print "Service clear_octomap call failed: %s"%e	

    
    
   
    ### Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(ros_msg) 

    ###  Statistical Outlier Filtering
    statistical_outlier_filter = pcl_data.make_statistical_outlier_filter()
    statistical_outlier_filter.set_mean_k(10)
    statistical_outlier_filter.set_std_dev_mul_thresh(0.01)
    pcl_data = statistical_outlier_filter.filter()

    ###  Voxel Grid Downsampling
    voxel_grid_filter = pcl_data.make_voxel_grid_filter()
    voxel_grid_filter.set_leaf_size(0.01, 0.01, 0.01)
    pcl_data = voxel_grid_filter.filter()

    ### PassThrough Filter Z
    passthrough_filter = pcl_data.make_passthrough_filter()
    filter_axis = 'z'
    passthrough_filter.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough_filter.set_filter_limits(axis_min, axis_max)
    pcl_data = passthrough_filter.filter()

    ### PassThrough Filter Y
    passthrough_filter = pcl_data.make_passthrough_filter()
    filter_axis = 'y'
    passthrough_filter.set_filter_field_name(filter_axis)
    axis_min = -0.4
    axis_max = 0.4
    passthrough_filter.set_filter_limits(axis_min, axis_max)
    pcl_data = passthrough_filter.filter()

    ### RANSAC Plane Segmentation
    segmenter = pcl_data.make_segmenter()    
    segmenter.set_model_type(pcl.SACMODEL_PLANE)
    segmenter.set_method_type(pcl.SAC_RANSAC)    
    segmenter.set_distance_threshold(0.01)
    inliers, coefficients = segmenter.segment()
    # Extract inliers and outliers
    pcl_data_objects = pcl_data.extract(inliers, negative=True)
    pcl_data_table = pcl_data.extract(inliers, negative=False)
  
    ### Euclidean Clustering
    pcl_data_objects_xyz = XYZRGB_to_XYZ(pcl_data_objects)
    tree = pcl_data_objects_xyz.make_kdtree()
    EuclideanClusterExtraction = pcl_data_objects_xyz.make_EuclideanClusterExtraction()
    EuclideanClusterExtraction.set_ClusterTolerance(0.05)
    EuclideanClusterExtraction.set_MinClusterSize(100)
    EuclideanClusterExtraction.set_MaxClusterSize(2000)
    # Search the k-d tree for clusters
    EuclideanClusterExtraction.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = EuclideanClusterExtraction.Extract()

    ### Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([pcl_data_objects_xyz[indice][0],
                                             pcl_data_objects_xyz[indice][1],
                                             pcl_data_objects_xyz[indice][2],
                                             rgb_to_float(cluster_color[j])])

    pcl_data_objects_clustered = pcl.PointCloud_PointXYZRGB()
    pcl_data_objects_clustered.from_list(color_cluster_point_list)

    ### Convert PCL data to ROS messages
    ros_objects_msg = pcl_to_ros(pcl_data_objects_clustered)
    ros_table_msg = pcl_to_ros(pcl_data_table)

    ### Publish ROS messages
    pcl_objects_pub.publish(ros_objects_msg)
    pcl_table_pub.publish(ros_table_msg)
    #TODO
    collidable_pub.publish(ros_table_msg)

    detected_objects = []
    detected_objects_labels = []
    ### Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_data_single_object_clustered = pcl_data_objects.extract(pts_list)
        ros_data_single_object_clustered = pcl_to_ros(pcl_data_single_object_clustered)
        # Compute the associated feature vector
        chists = compute_color_histograms(ros_data_single_object_clustered,using_hsv=True)
        normals = get_normals(ros_data_single_object_clustered)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # Publish a label into RViz
        label_pos = list(pcl_data_objects_xyz[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_data_single_object_clustered
        detected_objects.append(do)
	collidable_pub.publish(ros_data_single_object_clustered)

    ### Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(detected_object_list):
    # Initialize variables
    test_scene_num = Int32()
    #TODO GET THIS VALUE
    test_scene_num.data = 3 
    arm_name = String() 
    object_name = String() 
    pick_pose = Pose()
    place_pose = Pose()
    # Get/Read parameters
    pick_object_list = rospy.get_param('/object_list')
    dropbox_list = rospy.get_param('/dropbox')
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    dict_list=[]
    ### Loop through the pick list
    for pick_object in pick_object_list:
        print("Search Pick: ",pick_object['name'])
	for detected_object in detected_object_list:		
		print("Current detected_object: ",detected_object.label)
		if pick_object['name'] == detected_object.label:		
			object_name.data=pick_object['name']
			print("Pick: ",pick_object['name'])

			#TODO
			for detected_object_colliable in detected_object_list:
				if detected_object_colliable.label!=detected_object.label:
					print("Coolliable: ",detected_object_colliable.label)
					collidable_pub.publish(detected_object_colliable.cloud)

			# Calculate the centroid
			centroid=np.mean(ros_to_pcl(detected_object.cloud).to_array(), axis=0)[:3]
			# Fill up the position for the pick object
			pick_pose.position.x=np.asscalar(centroid[0])
			pick_pose.position.y=np.asscalar(centroid[1])
			pick_pose.position.z=np.asscalar(centroid[2])
			# Find the box to drop the object into.
			for dropbox in dropbox_list:
				if dropbox['group']==pick_object['group']:
					arm_name.data=dropbox['name']	
					dropbox_position = dropbox['position']
					break
			# Fill up the box position
			place_pose.position.x=dropbox_position[0]
			place_pose.position.y=dropbox_position[1]
			place_pose.position.z=dropbox_position[2]
			# Log 
			rospy.loginfo('Pick object {} with arm: {} and position [{},{},{}]'.format(object_name.data,arm_name.data,centroid[0],centroid[1],centroid[2]))
			# Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
			yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
			dict_list.append(yaml_dict)
			# Wait for 'pick_place_routine' service to come up
			rospy.wait_for_service('pick_place_routine')
			try:
			    pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
			    # Insert your message variables to be sent as a service request
			    resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
			    print ("Response: ",resp.success)
			    
			except rospy.ServiceException, e:
			    print "Service call failed: %s"%e	

			return
    # Output your request parameters into output yaml file
    send_to_yaml("dict_list.yaml",dict_list)

if __name__ == '__main__':
    # ROS node initialization
    rospy.init_node('clustering',anonymous=True)
    #TODO
    
    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points",PointCloud2, pcl_callback, queue_size=1)
    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table",PointCloud2, queue_size=1)
    collidable_pub = rospy.Publisher("/pr2/3d_map/points",PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers",Marker,queue_size=1)
    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    # Initialize color_list
    get_color_list.color_list = []
    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
