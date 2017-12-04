# Project: Perception Pick & Place

[//]: # (Image References)

[image1]:  ./misc_images/exercise-1-tablepod.png
[image2]:  ./misc_images/exercise-1-voxgrid.png
[image3]:  ./misc_images/exercise-1-passthrough.png
[image4]:  ./misc_images/exercise-1-inliers.png
[image5]:  ./misc_images/exercise-1-outliers.png
[image6]:  ./misc_images/exercise-2-gazebo.png
[image7]:  ./misc_images/exercise-2-rviz.png
[image8]:  ./misc_images/exercise-3-caputre-features.png
[image9]:  ./misc_images/exercise-3-training-svm.png
[image10]: ./misc_images/exercise-3-object-recognition.png
[image11]: ./misc_images/project-training-svm.png
[image12]: ./misc_images/objects-world-1.png
[image13]: ./misc_images/objects-world-2.png
[image14]: ./misc_images/objects-world-3.png




## Pipeline for filtering and RANSAC plane fitting implemented.

In the first exercise a Point Cloud for a tabletop with several object on top is given. Several filters are applied to finally generate a new Point Cloud for the objects and the tabletop separately. 

![alt text][image1]

The first filter is a Vox Grid filter in order to downsample the Point Cloud. This will decrease the number of elements to process in the next filters. After playing around a leaf size of 1 cm has been choosen.

```python
###### Voxel Grid filter
# Create a VoxelGrid filter object for our input point cloud
voxel_grid_filter = cloud.make_voxel_grid_filter()
# Set the voxel (or leaf) size  
voxel_grid_filter.set_leaf_size(0.01, 0.01, 0.01)
# Call the filter function to obtain the resultant downsampled point cloud
cloud_filtered = voxel_grid_filter.filter()
```

![alt text][image2]


The next filter is a pass through filter to cut off values outside a given range. In this case the Z axe is used, to eliminate the Point Cloud in the scene thar are not the tabletop itself or the objects on the tabletop.

```python
###### PassThrough filter
# Create a PassThrough filter object.
passthrough_filter = cloud_filtered.make_passthrough_filter()
# Assign axis and range to the passthrough filter object.
filter_axis = 'z'
passthrough_filter.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.1
passthrough_filter.set_filter_limits(axis_min, axis_max)
# Finally use the filter function to obtain the resultant point cloud. 
cloud_filtered = passthrough_filter.filter()
```

![alt text][image3]

Finally, a RANSAC filter is used to detect the tabletop. A fit with a plane model is used, since the tabletop has got a shape similar to a plane. This will allow to separate the tabletop, the inliers, from the objects on the table, the outliers.

```python
###### RANSAC plane segmentation
# Create the segmentation object
segmenter = cloud_filtered.make_segmenter()
# Set the model you wish to fit 
segmenter.set_model_type(pcl.SACMODEL_PLANE)
segmenter.set_method_type(pcl.SAC_RANSAC)
# Max distance for a point to be considered fitting the model
max_distance = 0.01
segmenter.set_distance_threshold(0.01)
# Call the segment function to obtain set of inlier indices and model coefficients
inliers, coefficients = segmenter.segment()
# Extract inliers
extracted_inliers = cloud_filtered.extract(inliers, negative=False)
# Extract outliers
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
```

![alt text][image4]

![alt text][image5]

## Pipeline including clustering for segmentation implemented.

In the second exercise a ros node written in python has been done. This node subscribes to the topic /sensor_stick/point_clould in order to receive a Point Clould similar to the previous exercise, a tablepot with several objects on top. After appying several filters, the same than in the exercise 1, the inliers, the tabletop, is published into the /pcl_table topic. An additional step is done to apply a Euclidean Clustering for the outlier, the objects on the tabletop. A different color is given to every detected object in the scene. The result is published into /pcl_object topic.

### Creating the ros node

Creates a ros node, called clustering, to subscribe to the Point Cloud, coming from a gazebo, with a tablepod with object on top.


```python
if __name__ == '__main__':
    # ROS node initialization
    rospy.init_node('clustering',anonymous=True)
    # Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud",pc2.PointCloud2,pcl_callback,queue_size=1)
    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects",pc2.PointCloud2,queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table",pc2.PointCloud2,queue_size=1)
    # Initialize color_list
    get_color_list.color_list = []
    # Spin while node is not shutdown
    while not rospy.is_shutdown():
     rospy.spin()
```

![alt text][image6]

### Euclidean Clustering

Euclidean Cluster algorithm to segment the Point Cloud into individual objects. A diffent color per object is used to visualiza the result.


```python
# Euclidean Clustering
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
```

```python
# Create Cluster-Mask Point Cloud to visualize each cluster separately
cluster_color = get_color_list(len(cluster_indices))
color_cluster_point_list = []
for j, indices in enumerate(cluster_indices):
for i, indice in enumerate(indices):
    color_cluster_point_list.append([pcl_data_objects_xyz[indice][0],
                                     pcl_data_objects_xyz[indice][1],
                                     pcl_data_objects_xyz[indice][2],
                                     rgb_to_float(cluster_color[j])])

```

![alt text][image7]


## Features extracted and SVM trained. Object recognition implemented.

In this exercise I am going to train a Support Vector Machine to recognize objects in a scene. A gazebo world containing the tablepod and the objects on top, the same than in the previous exercies, is provided. From these objects some features are extracted: histogram for both the HSV colorspace and the normal's surface. These extract features are used to train the SVN classifier, finally the classifier will predict what objects are in a segmented Point Cloud.

### Generating Features

For every object in the training scene its features are extracted. At the end the information is safe into the training_set.sav file. The number of different poses for the object have been increase to one hundred. Finally, the HSV colorspace is used, rather than the RBG, because it behaviour is better when the ilumination of the scene change.

```python
labeled_features = []
for model_name in models:
	spawn_model(model_name)

	for i in range(100):
	    # make five attempts to get a valid a point cloud then give up
	    sample_was_good = False
	    try_count = 0
	    while not sample_was_good and try_count < 5:
		sample_cloud = capture_sample()
		sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

		# Check for invalid clouds.
		if sample_cloud_arr.shape[0] == 0:
		    print('Invalid cloud detected')
		    try_count += 1
		else:
		    sample_was_good = True

	    # Extract histogram features
	    chists = compute_color_histograms(sample_cloud, using_hsv=True)
	    normals = get_normals(sample_cloud)
	    nhists = compute_normal_histograms(normals)
	    feature = np.concatenate((chists, nhists))
	    labeled_features.append([feature, model_name])

delete_model()
pickle.dump(labeled_features, open('training_set.sav', 'wb'))
```

![alt text][image8]

### Train the SVM

After the features have been generated the SVM will be trained. The result is safe into the model.sav file.

![alt text][image9]


### Object recognition

The previously SVM trained is used to recognize the object on the scene. The ros node created in the previous exercise is modified to include the SVM prediction.

```python
detected_objects = []
detected_objects_labels = []
# Classify the clusters! (loop through each detected cluster one at a time)
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

# Publish the list of detected objects
rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
detected_objects_pub.publish(detected_objects)
```

![alt text][image10]


## Pick and Place Setup

The implementation of the project include: to recognize the objects on the tablepod, to place them in the bin without piling up them, to rotate the robot to create a 3D collision map of the scene. The following sections will explain how this goals have been achive.

### Recognize the Objects

This part of the project have been done using the same filters than in the previous exercise. In order to train the SVM the script capture_features.py have been modified to include the objects for the three different worlds.

```python
models = [\
	    'biscuits',
            'soap',
            'soap2',
            'book',
            'glue',
            'sticky_notes',
            'snacks',
            'eraser']
``` 

The SVM was trained with this model and the result was:

![alt text][image11]

The PR2 Pick and Place simulator was executed with the three different tabletop configurations. The following images show how the objects in the scene were recongnized:

![alt text][image12]
![alt text][image13]
![alt text][image14]


### Place in the bin withou piling up

A new parameter has been included in the pick_list_x.yaml file, offset, in order to not place the object in the same position of the bin. This offset is added to X axes of the pose location.

```python
# Fill up the box position
place_pose.position.x=dropbox_position[0]+pick_object['offset'] # Include an offset to place the obejct in differents parts of the bin
place_pose.position.y=dropbox_position[1]
place_pose.position.z=dropbox_position[2]

```

### Create a 3D collision map 

Everytime a new Point Cloud is received a call to the clear_octomap is done in order to start with a new a fresh collision map.

```python
    def clear_octomap(self):
        rospy.wait_for_service('/clear_octomap')
        try:
            clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
            clear_octomap()
            print ("Response: OK" )     
        except rospy.ServiceException, e:
            print "Service clear_octomap call failed: %s"%e

```

The robot is rotated to the left and then to the right, the Point Cloud is analysed with the filters in order to extract the tabletop and the result is store into a list. Finally, these Points Cloud together with the objects we are not picking up in this iteration are sent to the /pr2/3d_map/point topic.


```python
    self.position_to_move_robot= [-1.7,0,1.7,0]

    def move_robot(self,fvalue):
        value = Float64()
        value.data=fvalue
        world_joint_pub.publish(value)
        if np.fabs(self.state_subcriber.get_joint_position('world_joint')-fvalue) < 1e-3:
            self.position_to_move_robot_index+=1

```

```python
# Add the table to the collidable objects. It stores the previous point cloud
# ir order to publish this values as the robot rotate around the scene
self.collidable_to_pub_list.append(pcl_data_table)
for collidable_to_pub in self.collidable_to_pub_list:
    ros_collidable_to_pub = pcl_to_ros(collidable_to_pub)
    collidable_pub.publish(ros_collidable_to_pub)
```

```python
# Add to the collidable publisher the objects that are not going to be collect in this run  
for detected_object_colliable in detected_object_list:
    if detected_object_colliable.label!=detected_object_found.label:
        print("detected_object_colliable=", detected_object_colliable.label)
        collidable_pub.publish(pcl_to_ros(detected_object_colliable.cloud))

```


## Conclusion

The recognition algorithm, using the filter, worked quite well. Also, the robot was able to make the 3D collision map, after rotating about itself, satisfactorily. I found a problem when the offset to place the object in the bin was to high, the navigation plan was not able to calculate the trajectory; I think it happens due to the arm was not able to get to this position without rotation the robot. The performance of the script was poor, I think I will need to run the script in a powerfull computer or to completly write it in C++ rather than python.


