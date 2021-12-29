import os

# Set folderpath to the folder containing the images to rename
# Takes a list of jpg files from folderpath and renames
# them from original name to prefix_1.jpg .. prefix_n.jpg
# where n is the number of images in the folderpath
folderpath = r'$HOME/Documents/Basic_Hair_Modified/val/T3'
prefix = 'T3'
filenumber = 1

for filename in os.listdir(folderpath):
    os.rename(folderpath + '/' + filename, folderpath + '/' + prefix + '_' + str(filenumber) + '.jpg')
    filenumber += 1