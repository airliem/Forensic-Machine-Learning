import albumentations as A
import cv2
import os

# Set folderpath to the folder containing the images 
# Set folderpath to the folder to contain the transformed images
# Takes a list of jpg files from folderpath and transforms each image
# three times and sends three transformed images to new folderpath
input_folder = r'$HOME/Documents/Basic_Hair_Modified/train/T3/'
output_folder = r'$HOME/Documents/Basic_Hair_Modified/train/T3a/'

def transform (file):
    inputfile = input_folder + file 
    image = cv2.imread(inputfile) 
    transform_1 = A.Compose([A.HorizontalFlip(p=1.0),]) #p=1.0 = all images
    transform_2 = A.VerticalFlip(p=1.0)
    transform_3 = A.RandomRotate90(p=1.0)
    augmented_image_a = transform_1(image=image)['image']
    augmented_image_b = transform_2(image=image)['image']
    augmented_image_c = transform_3(image=image)['image']
    outputfile = output_folder + "ta" + file 
    cv2.imwrite(outputfile, augmented_image_a) 
    outputfile = output_folder + "tb" + file 
    cv2.imwrite(outputfile, augmented_image_b) 
    outputfile = output_folder + "tc" + file 
    cv2.imwrite(outputfile, augmented_image_c) 

files = os.listdir(input_folder)
for file in files: 
    transform(file)