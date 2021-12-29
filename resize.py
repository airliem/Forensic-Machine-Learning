from PIL import Image
import os

# Set folderpath to the folder containing the images to resize
# Set folderpath to the folder to contain the resized images
# Takes a list of jpg files from folderpath and resizes
# them maintaining aspect ratio and sends resized images to 
# new folderpath
input_folder = r'$HOME/Documents/Basic_Hair_Modified/train/T3/'
output_folder = r'$HOME/Documents/Basic_Hair_Modified/train/T3a/'

def resize(file, new_width): 
    inputfile = input_folder + file 
    im = Image.open(inputfile) 
    width, height = im.size 
    ratio = height/width 
    new_height = int(ratio * new_width) 
    resized_image = im.resize((new_width, new_height)) 
    outputfile = output_folder + "resize" + file 
    resized_image.save(outputfile) 

files = os.listdir(input_folder)

for file in files:
    resize(file, 256) #256
    print(type(file))