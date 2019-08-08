# CNN_for_TEM_Segmentation
Python scripts for data set preparation and CNN training/inference

CNN_models: jupyter notebooks with examples of CNN model design and training procedures
   -HighRes - 4 convolutional steps with batch norm and second convolutional layer at each step (commented out)
   -Original - 3 layer CNN as used on 512x512 images
   -UNet_Leaky... - same architecture as HighRes, but with Leaky ReLU and learning rate scheduler
   
dataset_prep: 
   -job.sh - bash script for parallel processing of images
   -label_training_set - example of applying filters, reconstruction to raw ETEM images
   -training_set_augmentation - example script for augmenting images
   
training_data:
   -sample_training_images/labels - set of 15 images/labels in a numpy array for processing by scripts in dataset prep
   ***Note: processing using job.sh, etc. uses jpg images; dataset_prep scripts can be easily modified to loop through a stack of images
   
