from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import sys
import os
import numpy as np
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) #we are initializing the model to use
        self.tracker = sv.ByteTrack()
        
    
    
    def detect_frames(self, frames):
        
        batch_size = 20 #we'll use it to pass chunks of frames instead of the whole thing at once
        
        detections = []
        
        for i in range(0, len(frames), batch_size): #this loop starts from  0 and each iteration increments by batch_size
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.1) #0.1 is the minimum acceptable confidence
            detections += detections_batch
            
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path): #checks if there is a file to read rather than processing all the frames
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        
        #creates an object 
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }
        
        for frame_num, detection in enumerate(detections): #loops through the detections 
            #the enumerate function call puts in the index from the list 
            
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            #convert to supervision detection 
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #converts any found goalkeeper tag to a player tag.
            #we can do this since we aren't doing any sort of special stat with the goalies 
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]                    
            
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            #cretes an empty dictionary for each tag 
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist() #we pick 0 since that is the index at which the is
                class_id = frame_detection[3] #same concept here as in why we select the third index 
                track_id = frame_detection[4]
                
                if class_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                    
                if class_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                
                if class_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)
        return tracks
    
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        
        y2 = int(bbox[3])
        
        x_center, _ = get_center_of_bbox(bbox)  #we use _ to show that the  y center will not be used
        
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame,
            center= (x_center, y2),
            axes= (int(width), int(0.35 * width)),
            angle= 0.0,
            startAngle= -45, #this means that the ellipse will be drawn from angles 45 up to 235
            endAngle= 235,
            color= color,
            thickness= 2,
            lineType= cv2.LINE_4
            
        )
        
        rectangle_width = 40
        rectangle_height = 20
        
        #x cords
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        
        #y cords
        y1_rect = (y2 - rectangle_height//2) + 15   #the +15 is just for padding 
        y2_rect = (y2 + rectangle_height//2)  + 15 
        
        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)),(int(x2_rect),int(y2_rect)), color, cv2.FILLED)
            
            
            x1_text = x1_rect+12
            
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)), #the 15 is for padding 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, # size 
                (0,0,0), #text color
                2 #thickness
            ) 
        
        
        return frame
        
        
    
    
    def draw_triangle(self,frame, bbox, color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x,y], 
            [x-10,y-20],
            [x+10,y-20]
        ])
        
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) #TRIANGLE 
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) #TRIANGLE BORDER
        
        return frame
    
    #now lets turn the bounding boxes into nice circles 
    def draw_annotations(self, video_frames, tracks):
        
        output_video_frames = [] #array where frames will be stored 
        
        for frame_num, frame, in enumerate(video_frames):
            #inside of this loop is where we are going to draw the circles around players
            
            frame = frame.copy() #we do this so that we don't modify the original frame list but rather a copy of it
            
            
            #dictionaries of the players, ball and refs for the given frame 
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            
            
            #draw players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame,player["bbox"],(75,0,130), track_id)
            
            #draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee["bbox"],(0,255,255))
                
            
            #draw ball 
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))
                
            output_video_frames.append(frame)
        
        return output_video_frames
    