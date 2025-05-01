def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    
    #we need to find the center of it
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]