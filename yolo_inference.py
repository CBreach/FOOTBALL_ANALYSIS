#this is a file that will be used to get to know yolo
#i've never worked with this before so, it'll help me 
#get familiar with the library 

from ultralytics import YOLO

#shootout this was the original name of the vid 
#model = YOLO('yolov8x') #this loads the model --> we are using the 8th version

model = YOLO('models/best.pt') #now that we trained the model we can load it instead of the default one

output = model.predict('input_videos/input_1.mp4', save = True) #this loads the input video into the model and saves the output

print(output[0]) #index 0 is the first frame of the processed video 
print("###################")

for box in output[0].boxes:
    print(box)

