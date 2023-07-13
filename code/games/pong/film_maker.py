import cv2
import glob
import os 
from tqdm import tqdm 

# Specify video parameters
img_array = []
frame_size = None
fps = 30.0

# Read the frames in the correct order
files = sorted(glob.glob('./movie/*.png'))
for filename in tqdm(files):
    img = cv2.imread(filename)
    img = cv2.flip(img, 0)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    height, width, layers = img.shape
    frame_size = (width, height)
    img_array.append(img)

# Specify the video codec, create the VideoWriter and write the frames to it
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./movie/project.mp4', fourcc, fps, frame_size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()

for file in files:
    os.remove(file)
