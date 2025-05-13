
"""
Self-training process. 
We use a simple classification. A pre-trained yolo classifier. Install Ultralytics by following the instructions from https://docs.ultralytics.com/
If you have imbalanced classes, consider using the Weighted Loss Function. 

"""
############################################### 1: TRAIN ON LABELLED FEW SAMPLES  #################################################################
# Import required libraries 
from ultralytics import YOLO

import os
os.chdir("workingDir")  
# Load a model
model = YOLO('YOLOv8x-cls.yaml')  # build a new model from YAML
#model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
model = YOLO('YOLOv8x-cls.yaml').load('add pretrained model path here')  # build from YAML and transfer weights

# Train the model using images with the first three bands
results = model.train(data='./trainSetOfFewLabelledRoofCrops', epochs=300, imgsz=224) # the default size is 224

metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.top1   # top1 accuracy
metrics.top5   # top5 accuracy

##################################################### 2. LEVERAGE PREDICTION WITH HIGH CONFIDENCE TO EXPDAND THE TRAINING SET ########################
#make classification and write result to csv file 
import os
import csv
import numpy as np
from PIL import Image

# Load the best model
model = YOLO('./bestModel.pt')
model.to("cpu")

# Define classes names
classes_names = ["1", "2"]

# Define folder containing images
folder_path = "./allsampleRoofCrops"

# Define CSV file path to save results
csv_file_path = "./pseudolabels.csv"

# Open CSV file for writing
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['image', 'Class', 'Probability']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate through each image file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".tif") or filename.endswith(".png"):
            # Predict with the model
            img_path = os.path.join(folder_path, filename)
            results = model(img_path)

            # Process predictions
            for each in results:
                probabilities = each.probs.numpy().data
                class_index = probabilities.argmax()  # Get the index with the highest prob. which class is likely?
                class_name = classes_names[class_index]  # Map the index to the class name
                probability = probabilities[class_index] 

                # Save results to CSV
                writer.writerow({'image': filename, 'Class': class_name, 'Probability': probability})
                              

#Create a  new training dataset containing pseudo labels  which confidence is very high 
import os
import shutil
import pandas as pd

# Function to create subsets and move images based on class
def create_class_subsets(df, source_dir, target_dir):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Create 2 subfolders (1 for High,  2 for Low)
    for class_name in [1, 2]:
        class_folder = os.path.join(target_dir, str(class_name))
        os.makedirs(class_folder, exist_ok=True)
    
    # Iterate over the DataFrame and copy images to respective subfolders
    for _, row in df.iterrows():
        image_name = row['image']
        image_class = row['Class']
        
        # Define the source and target paths
        src_path = os.path.join(source_dir, image_name)
        dest_path = os.path.join(target_dir, str(image_class), image_name)
        
        # Check if the image exists in the source directory and copy it
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Copied {image_name} to class {image_class} folder.")
        else:
            print(f"Image {image_name} not found in source directory.")


# Example usage
df = pd.read_csv('C:/dl/yolo/selftraining/sample_class_wealth_sample.csv')
df=df[df['Probability']>=0.9] 
source_directory = 'C:/dl/yolo/selftraining/sat2022'  # Source folder where the images are stored
target_directory = 'C:/dl/yolo/selftraining/student1sat'  # Target folder where the balanced dataset will be created

create_class_subsets(df, source_directory, target_directory)

"""
retrain model as in 1 and repeate the process 3-4 times until maximum of samples are classified with high accuracy. The final model will apply to all samples to geenrate the 
classified map of roofs
"""

########################### refine the dataframe with roofPatches ID and their assigned class and join this to orginal shapefile#############################
      
import pandas as pd

# Load the CSV file
file_path = "C:/dl/MT/Musanze/inferencenf.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Remove the .png extension from the "image" column
df['image'] = df['image'].str.replace('.png', '', regex=False) # remove the extension to remin with image ID to make join to shapefile possible

# Save the modified DataFrame back to a CSV file
output_path = 'C:/dl/MT/Musanze/modified_inferencenf.csv'  # Replace with your desired output file path
df.to_csv(output_path, index=False)

print(f"Modified file saved to {output_path}")
