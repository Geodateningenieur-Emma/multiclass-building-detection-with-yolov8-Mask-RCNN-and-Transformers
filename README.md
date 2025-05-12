# Housing wealth mapping using multiclass-building-detection -with-YOLO and-Mask2Former

Buildings can vary significantly in size, shape, material, or usage and values.
Multi-class building detection refers to the process of detecting and classifying buildings into multiple categories, typically based on their attributes or characteristics. 
This is an advanced step beyond binary building detection and has several practical applications, particularly in areas like urban planning, and development assessemnt.

our implmenetation follow the following steps 

1. Data preparation
   Praparing annotation data include several steps:
   crop roof patches which were sent to experts via online google form for annotation.
   The code loop through all building polygon and use the geometry to crop the image to the extent of the roof.
   Each building in the building shapefile was assigned a unique identifier (bID) that allow to track it and join its class from experts to original shapefile. 
   experts were tasked to tell which wealth class the building likely represents (1: Low, 2: High). 
   Self-training leveraging fews experts annotations. To labels all the buildings in our training set
   (it is impossible to make a google form with tens thousands of images patches) a YOLO classifier pretrained on Imagenet was used to generate pseudo labels.
   we selected only predictions with a confidence of 0.9. we run our self-training with 3 itertations and the final model was applied to albels all buildings in the sample area. we predicted the wealth class for each roof crop and save results to csv providing details of each crop(bID, class) and then join
4. Training multi-class models which identify and delineate building and add a wealth annotation (low or high)
5. Binning predictions in grids of 100x100 meter

