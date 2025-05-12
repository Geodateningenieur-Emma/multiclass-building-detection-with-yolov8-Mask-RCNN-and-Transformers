# Housing wealth mapping using multiclass-building-detection -with-YOLO and-Mask2Former

Buildings can vary significantly in size, shape, material, or usage and value.
Multi-class building detection refers to the process of detecting and classifying buildings into multiple categories, typically based on their attributes or characteristics. 
This is an advanced step beyond binary building detection and has several practical applications, particularly in urban planning and development assessment.

Our implementation follows the following steps 

1. [Preparing annotation](https://github.com/Geodateningenieur-Emma/multiclass-building-detection-with-yolov8-Mask-RCNN-and-Transformers/tree/main/prepare%20annotation)  
   This includes several steps:  
   - Crop roof patches, which were sent to experts via an online Google form for annotation.
   The [code](https://github.com/Geodateningenieur-Emma/multiclass-building-detection-with-yolov8-Mask-RCNN-and-Transformers/blob/main/prepare%20annotation/1.%20Get%20roof%20crops.py) loops through all building polygons and uses the geometry to crop the image to the extent of the roof.
   Each building in the building shapefile was assigned a unique identifier (bID) that allows tracking it and joining its class from experts to the original shapefile. 
   experts were tasked to tell which wealth class the building likely represents (1: Low, 2: High).   
   - Self-training leveraging a few experts' annotations. To label all the buildings in our training set
   (It is impossible to make a Google form with tens of thousands of image patches). We used this [code](https://github.com/Geodateningenieur-Emma/multiclass-building-detection-with-yolov8-Mask-RCNN-and-Transformers/blob/main/prepare%20annotation/2.%20Self-Training.py): a YOLO classifier pre-trained on Imagenet was used to generate pseudo labels, selecting only candidates with a confidence of 0.9, for 3 iterations. The final model was applied to label all buildings in the sample area.  We predicted the wealth class for each roof crop and saved the results to a CSV, providing details of each crop(bID, class), and then joined it to the shapefile. 
 - Converting a shapefile to the multiclass gray image. The [code](https://github.com/Geodateningenieur-Emma/multiclass-building-detection-with-yolov8-Mask-RCNN-and-Transformers/blob/main/prepare%20annotation/3.%20Shapefile2Multiclass%20grey%20image%20patches.py) rasterises the vector building polygons and creates label raster of the same dimension as original image (e.g UAV, aerial image that need to annotate). Use GDAL Command Line Interface (CLI) to generate patches (of both original  and label). original images and corresponding classified building label images were then converted to [YOLO-ANNOTATION](https://github.com/Geodateningenieur-Emma/multiclass-building-detection-with-yolov8-Mask-RCNN-and-Transformers/blob/main/prepare%20annotation/4.1.%20LabeledMaskImageAnnotation2YoloFormat.py) and [Mask2Former-ANNOTATION](https://github.com/Geodateningenieur-Emma/multiclass-building-detection-with-yolov8-Mask-RCNN-and-Transformers/blob/main/prepare%20annotation/4.2.%20grey%20image%20to%20classified%20image%20compatible%20to%20mask2former.py)

2. Training and inferencing:
   - First, we installed [YOLO from Ultralytics](https://docs.ultralytics.com/de/quickstart/) and [Mask2Former](https://arxiv.org/abs/2112.01527) following a [tutorial](https://debuggercafe.com/multi-class-segmentation-using-mask2former/) by Sovit Ranjan Rath (2024). This [code](https://github.com/Geodateningenieur-Emma/multiclass-building-detection-with-yolov8-Mask-RCNN-and-Transformers/blob/main/Training/TrainYOLO.py). The mask2former is trained by running the training script as described [here](https://github.com/Geodateningenieur-Emma/multiclass-building-detection-with-yolov8-Mask-RCNN-and-Transformers/blob/main/Training/Mask2Former).
   - Inference 

  
5. Binning predictions in grids of 100x100 meter

