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
from sensor_msgs.msg import JointState 
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import threading

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


class JointStateSubscriber:
    """ Helper class to subscribe to the joint state """
    def __init__(self):
        """ Subscribes to the joint states topic """
        self.lock = threading.Lock()
        self.subsriber = rospy.Subscriber("/joint_states",JointState,self.callback,queue_size=1)
        self.name = []
        self.position = []
        
    def callback(self,joint_state):
        """ Callback function for the topic. It stores the names and position of the joints """
        self.lock.acquire()
        self.name=joint_state.name
        self.position=joint_state.position
        self.lock.release()
        
    def get_joint_position(self,joint_name):
        """ Given a joint name it returns its position """
        if self.name==[]:
            return 0.0
        self.lock.acquire()
        index = self.name.index(joint_name)
        position = self.position[index]
        self.lock.release()
        return position
        
class PointCloudSubscriber:
    """ Subscriber to the Point Cloud """
    def __init__(self,JointStateSubscriber):
        """ Subscribe to the pr2 world points """
        self.subscriber = rospy.Subscriber("/pr2/world/points",PointCloud2, self.callback, queue_size=1)
        self.pcl_data = None
        self.collidable_to_pub_list = []
        self.position_to_move_robot_index=0
        self.state_subcriber=JointStateSubscriber
        self.position_to_move_robot= [-1.7,0,1.7,0]
            
    def callback(self,ros_msg):
        """ Callback function for the topic  """

        # Convert ROS msg to PCL data
        self.pcl_data = ros_to_pcl(ros_msg)        
        
        # Statistical Outlier Filtering
        self.pcl_data=self.statistical(self.pcl_data)
        
        # Voxel Grid Downsampling
        self.pcl_data=self.voxel_grid(self.pcl_data)
        
        # PassThrough Filter
        self.pcl_data=self.passthrough(self.pcl_data)
        
        # RANSAC Plane Segmentation
        pcl_data_table, pcl_data_objects = self.segmenter(self.pcl_data)
               
        # Convert PCL data table to ROS messages
        ros_data_table = pcl_to_ros(pcl_data_table)
               
        # Publish ROS data table messages
        pcl_table_pub.publish(ros_data_table)
        
        # Add the table to the collidable objects. It stores the previous point cloud
        # ir order to publish this values as the robot rotate around the scene
        self.collidable_to_pub_list.append(pcl_data_table)
        for collidable_to_pub in self.collidable_to_pub_list:
            ros_collidable_to_pub = pcl_to_ros(collidable_to_pub)
            collidable_pub.publish(ros_collidable_to_pub)

        # Rotate the Robot to detect all the borders of the table in the scene.        
        if self.position_to_move_robot_index < len(self.position_to_move_robot):
            self.move_robot(self.position_to_move_robot[self.position_to_move_robot_index])
            return
        
        # Euclidean Clustering
        cluster_indices, pcl_data_objects_clustered,pcl_data_objects_xyz = self.cluster(pcl_data_objects)
        
        # Convert PCL data object clustered to ROS messages
        ros_data_objects_clustered = pcl_to_ros(pcl_data_objects_clustered)
        
        # Publish ROS clustered object messages
        pcl_objects_pub.publish(ros_data_objects_clustered)

        # Classify the clusters! (loop through each detected cluster one at a time)
        detected_objects_labels,detected_objects = self.classifier(pcl_data_objects,cluster_indices,pcl_data_objects_xyz)
        
        rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

        # Publish the list of detected objects        
        detected_object_pub.publish(detected_objects)
        
        # Pick up the object 
        try:
            self.pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass

        # Clear the octomap to start with a fresh one.
        self.clear_octomap()           

    def pr2_mover(self,detected_object_list):
    
        # Initialize variables
        test_scene_num = Int32()
        test_scene_num.data = 1 
        arm_name = String() 
        object_name = String() 
        pick_pose = Pose()
        place_pose = Pose()
        dict_list=[]
        
        # Get/Read parameters
        pick_object_list = rospy.get_param('/object_list')
        dropbox_list = rospy.get_param('/dropbox')
        
        # Loop through the pick list
        for pick_object in pick_object_list:
            print("Search Pick: ",pick_object['name'])
            
            detected_object_found = None
            
            for detected_object in detected_object_list:        
                print("Current detected_object: ",detected_object.label)
                if pick_object['name'] == detected_object.label:
                    detected_object_found = detected_object
                    break
            
            if detected_object_found is not None:
                            
                object_name.data=pick_object['name']
                print("Pick: ",pick_object['name'])
                
                # Add to the collidable publisher the objects that are not going to be collect in this run  
                for detected_object_colliable in detected_object_list:
                    if detected_object_colliable.label!=detected_object_found.label:
                        print("detected_object_colliable=", detected_object_colliable.label)
                        collidable_pub.publish(pcl_to_ros(detected_object_colliable.cloud))

                ## Get the PointCloud for a given object and obtain it's centroid
                # Calculate the centroid
                centroid=np.mean(detected_object_found.cloud.to_array(), axis=0)[:3]
                # Create 'place_pose' for the object
                # Fill up the position for the pick object
                pick_pose.position.x=np.asscalar(centroid[0])
                pick_pose.position.y=np.asscalar(centroid[1])
                pick_pose.position.z=np.asscalar(centroid[2])

                ## Assign the arm to be used for pick_place
                # Find the box to drop the object into.
                for dropbox in dropbox_list:
                    if dropbox['group']==pick_object['group']:
                        arm_name.data=dropbox['name']    
                        dropbox_position = dropbox['position']
                        break

                # Fill up the box position
                place_pose.position.x=dropbox_position[0]+pick_object['offset'] # Include an offset to place the obejct in differents parts of the bin
                place_pose.position.y=dropbox_position[1]
                place_pose.position.z=dropbox_position[2]
                # Log 
                rospy.loginfo('Pick object {} with arm: {} and position [{},{},{}]'.format(object_name.data,arm_name.data,centroid[0],centroid[1],centroid[2]))
   
                ## Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
                yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
                dict_list.append(yaml_dict)

                # Wait for 'pick_place_routine' service to come up
                rospy.wait_for_service('pick_place_routine')

                try:
                    pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

                    # Insert your message variables to be sent as a service reques
                    resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
                    print ("Response: ",resp.success)

                except rospy.ServiceException, e:
                    print "Service call failed: %s"%e

                return
    
        # Output your request parameters into output yaml file
        send_to_yaml("dict_list.yaml",dict_list)
        
    def statistical(self,pcl_data):
        statistical_outlier_filter = pcl_data.make_statistical_outlier_filter()
        statistical_outlier_filter.set_mean_k(10)
        statistical_outlier_filter.set_std_dev_mul_thresh(0.1)
        pcl_data = statistical_outlier_filter.filter()
        return pcl_data
        
    def voxel_grid(self,pcl_data):
        voxel_grid_filter = pcl_data.make_voxel_grid_filter()
        voxel_grid_filter.set_leaf_size(0.01, 0.01, 0.01)
        pcl_data = voxel_grid_filter.filter()
        return pcl_data        
        
    def passthrough(self,pcl_data):
        passthrough_filter = pcl_data.make_passthrough_filter()
        filter_axis = 'z'
        passthrough_filter.set_filter_field_name(filter_axis)
        axis_min = 0.6
        axis_max = 1.1
        passthrough_filter.set_filter_limits(axis_min, axis_max)
        pcl_data = passthrough_filter.filter()
        return pcl_data

    def segmenter(self,pcl_data):
        segmenter = pcl_data.make_segmenter()    
        segmenter.set_model_type(pcl.SACMODEL_PLANE)
        segmenter.set_method_type(pcl.SAC_RANSAC)    
        segmenter.set_distance_threshold(0.01)
        inliers, coefficients = segmenter.segment()
        # Extract inliers and outliers
        pcl_data_objects = pcl_data.extract(inliers, negative=True)
        pcl_data_table = pcl_data.extract(inliers, negative=False)        
        return pcl_data_table, pcl_data_objects 
   
    def cluster(self,pcl_data_objects):
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
        
        ## Create Cluster-Mask Point Cloud to visualize each cluster separately
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
        
        return cluster_indices,pcl_data_objects_clustered,pcl_data_objects_xyz
        
    def classifier(self,pcl_data_objects,cluster_indices,pcl_data_objects_xyz):
        detected_objects = []
        detected_objects_labels = []
        ## Classify the clusters! (loop through each detected cluster one at a time)
        for index, pts_list in enumerate(cluster_indices):
            # Grab the points for the cluster
            pcl_data_single_object = pcl_data_objects.extract(pts_list)
            ros_data_single_object = pcl_to_ros(pcl_data_single_object)
            # Compute the associated feature vector
            chists = compute_color_histograms(ros_data_single_object,using_hsv=True)
            normals = get_normals(ros_data_single_object)
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
            do.cloud = pcl_data_single_object
            detected_objects.append(do)        
        return detected_objects_labels,detected_objects

    def clear_octomap(self):     
        rospy.wait_for_service('/clear_octomap')
        try:
            clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
            clear_octomap()
            print ("Response: OK" )     
        except rospy.ServiceException, e:
            print "Service clear_octomap call failed: %s"%e
            
            # function to load parameters and request PickPlace service
            
    def move_robot(self,fvalue):
        value = Float64()
        value.data=fvalue
        world_joint_pub.publish(value)
        if np.fabs(self.state_subcriber.get_joint_position('world_joint')-fvalue) < 1e-3:
            self.position_to_move_robot_index+=1
        


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering',anonymous=True)

    # Create Subscribers
    JointStateSubscriber = JointStateSubscriber()
    PointCloudSubscriber = PointCloudSubscriber(JointStateSubscriber)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table",PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers",Marker,queue_size=1)
    detected_object_pub = rospy.Publisher("/detected_objects",DetectedObjectsArray,queue_size=1)
    collidable_pub = rospy.Publisher("/pr2/3d_map/points",PointCloud2, queue_size=1)
    world_joint_pub = rospy.Publisher("/pr2/world_joint_controller/command",Float64,queue_size=1)

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
