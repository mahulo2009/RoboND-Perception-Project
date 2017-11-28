# Project: Perception Pick & Place

[//]: # (Image References)

[image1]: ./misc_images/exercise-1-tablepod.png
[image2]: ./misc_images/exercise-1-voxgrid.png
[image3]: ./misc_images/exercise-1-passthrough.png
[image4]: ./misc_images/exercise-1-inliers.png
[image5]: ./misc_images/exercise-1-outliers.png


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

In the second exercise a ros node written in python has been done. This node subscribe to the topic /sensor_stick/point clould in order to receive a Point Clould similar to the previous excersice, a tablepot with several objects on top. After appying several filters, the same 





### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



