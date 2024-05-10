
def _mkdir(folder):
    
    import os 
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    return []


def _read_obj_names(textfile):
    """ helper to read the class files for detection  
    """
    import numpy as np 

    classnames = []
    
    with open(textfile) as f:
        for line in f:
            line = line.strip('\n')
            if len(line)>0:
                classnames.append(line)
            
    return np.hstack(classnames)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3    

def _sigmoid(x):
    import numpy as np 
    return 1. / (1. + np.exp(-x))


def _bbox_iou(box1, box2):

    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union


# def preprocess_input(image, net_h, net_w):
    
#     import cv2
#     import numpy as np 
    
#     new_h, new_w, _ = image.shape

#     # determine the new size of the image
#     if (float(net_w)/new_w) < (float(net_h)/new_h):
#         new_h = (new_h * net_w)/new_w
#         new_w = net_w
#     else:
#         new_w = (new_w * net_h)/new_h
#         new_h = net_h

#     # resize the image to the new size
#     resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

#     # embed the image into the standard letter box
#     new_image = np.ones((net_h, net_w, 3)) * 0.5
#     new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
#     new_image = np.expand_dims(new_image, 0)

#     return new_image


class BoundBox:
    """ The class stores bounding boxes in voc format
    
    Sample Usage:
        bbox = BoundBox(xmin, ymin, xmax, ymax)
        
    """
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        """ This function gets the label with highest score for this box 
        """
        import numpy as np 
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        """ This function gets the score for the label of this box
        """
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score
    

def _decode_netout(netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
    """ Function to decode the output of the YOLOv3 score maps to parse the bounding boxes. 
    
    Sample Usage:
        bbox = BoundBox(xmin, ymin, xmax, ymax)
        
    """
    import numpy as np 
    
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            #objectness = netout[..., :4]
            
            if(objectness.all() <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            #box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, None, classes)

            boxes.append(box)

    return boxes

def _correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
        
def _do_nms(boxes, nms_thresh):
    import numpy as np 
    
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if _bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    

def _draw_bbox(image, detections, class_labels, cmap):
    
    # from skimage import io, draw
    import cv2
    import numpy as np
    # image = io.imread(imagePath)
    
    for detection in detections:
        label = str(detection[0])
        # lab_num = int(np.arange(len(class_labels))[class_labels==label])
        # box_color = (np.array(cmap[lab_num])*255).astype(np.uint8)
        
        confidence = float(detection[1])
        bounds = detection[2] #.astype(np.float)
        # shape = image.shape
        yExtent = int(bounds[3])
        xEntent = int(bounds[2])
        # Coordinates are around the center
        xCoord = int(bounds[0] - bounds[2]/2)
        yCoord = int(bounds[1] - bounds[3]/2)
        
        # print(tuple(box_color))
        # print((xCoord, yCoord), 
        #               (np.clip(xCoord + xEntent, 0, image.shape[1]), 
        #                np.clip(yCoord + yExtent, 0, image.shape[0])), 
        #               tuple(box_color), 3)
        pt1 = (xCoord, yCoord)
        pt2 = (np.clip(xCoord + xEntent, 0, image.shape[1]), np.clip(yCoord + yExtent, 0, image.shape[0]))
        cv2.rectangle(image, pt1, pt2, (0,255,0), 3)
        # print(box_color)
        # cv2.rectangle(image, pt1, pt2, (box_color[0],box_color[1],box_color[2]), 3)
        cv2.putText(image, 
                    label + ' %.1f' %(confidence*100), 
                    (np.clip(xCoord -13, 30, image.shape[1]), np.clip(yCoord + 13, 30, image.shape[0])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,0,255), 2)
    
    return image

def _get_bboxes(boxes, labels, obj_thresh):
    
    boxes_filt = []
    
    for box in boxes:
        label_str = ''
        label = -1
        conf = 0
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str += labels[i]
                label = i
                conf = box.classes[i]
                # print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
                
        if label >= 0:
            
            x1 = box.xmin
            x2 = box.xmax
            y1 = box.ymin
            y2 = box.ymax
            x = .5*(x1+x2)
            y = .5*(y1+y2)
            xEntent = x2-x1
            yEntent = y2-y1
            bbox_detect = [label_str, conf, [x, y, xEntent, yEntent]]
            boxes_filt.append(bbox_detect)
            
            # cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
            # cv2.putText(image, 
            #             label_str + ' ' + str(box.get_score()), 
            #             (box.xmin, box.ymin - 13), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 
            #             1e-3 * image.shape[0], 
            #             (0,255,0), 2)
    # if len(boxes_filt) > 0:
    #     boxes_filt = np.vstack(boxes_filt)
        
    return boxes_filt
        
    
def write_bbox_files(filename, detections):
    """ write bounding box detections into file 

    Parameters
    ----------
    filename : str 
        filepath to save bounding boxes to 
    detections : list of [label, score, box] bounding box detections
        box can be in VOC (x1,y1,x2,y2) or YOLO (x,y,w,h) format. 

    Returns
    -------
    None :

    """
    import numpy as np 
    with open(filename, 'w') as f:
        for det in detections:
            label, score, box = det
            box = np.array(box).astype(np.int32)
            
            f.write(label+'\t'+str(score)+'\t'+str(box[0])+'\t'+str(box[1])+'\t'+str(box[2])+'\t'+str(box[3])+'\n')
            
    return []


def load_and_run_YOLOv3_weights_keras_detection_model_RGB(vid, 
                                                          weightsfile,
                                                          outfolder,  
                                                          equalize_hist=False,
                                                          obj_thresh=0.001, 
                                                          nms_thresh = 0.45,
                                                          imsize=(512,512),  
                                                          anchors=None,
                                                          class_name_file=None,
                                                          debug_viz=False):
    """ Master function to predict all bounding boxes for every channel of an input RGB video given the YOLOv3 weights file and the output outfolder path for saving the resultant bounding box predictions. 
    
    Parameters
    ----------
    vid : (n_frames, n_rows, n_cols, 3) array 
        input RGB video.
    weightsfile : str
        filepath to the pretrained YOLOv3 weights saved in keras .h5 format. 
    outfolder : str
        folderpath to save the bounding box predictions. subfolders will be created in this folder for each of the channels and for each channel there will be an individual .txt file containing the bounding boxes at each timepoint
    equalize_hist : bool
        if True, apply global histogram equalization to each frame of the video. 
    obj_thresh : float [0-1], optional
        the score threshold for a positive bounding box detection. The default is 0.001.
    nms_thresh : float [0-1], optional
        the overlapping IoU threshold for non-maximum bounding box suppression to remove overlapping boxes. The default is 0.45.
    imsize : (n_rows, n_cols) tuple, optional
        the input image size to the trained YOLOv3 architecture. The default is (512,512).
    anchors : list of 3 lists, optional
        these specify the anchors giving the aspect ratio at each of the 3 scales in YOLOv3 architecture. The default is None.
    class_name_file : filepath, optional
        If supplied, it is a file where each line is the name of each object class. Here we have one class - 'organoid'. The default is None.
    debug_viz : bool, optional
        If True, draw the bounding box on the video frame for every video frame for checking. The default is False.

    Returns
    -------
    None.

    """
    from tensorflow.keras.models import load_model 
    import skimage.transform as sktform 
    import numpy as np 
    import skimage.exposure as skexposure 
    import os 
    import skimage.io as skio 
    import seaborn as sns 
    
    net_h, net_w = imsize # a multiple of 32, the smaller the faster
    if anchors is None:
        anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
    
    if class_name_file is None: 
        bbox_class_names = np.hstack(['organoid'])
    else:
        bbox_class_names = _read_obj_names(class_name_file)
    class_labels = bbox_class_names

    yolo_model = load_model(weightsfile)
    # aug_labels = np.insert(class_labels, len(class_labels), 'Normal')

    cmap = sns.color_palette('hls', len(class_labels)+1)
    
    for channel in range(vid.shape[-1])[:]:

        pred_folder = os.path.join(outfolder, 'Channel-%s' %(str(channel+1).zfill(2)))
        _mkdir(pred_folder) # create the folder for the channel. 

        # iterate over channel. 
        # imglist = []
        # vid_ch = equalize_hist(vid[...,channel].ravel()).reshape(vid[...,channel].shape) # 0-1
        # vid_ch = vid[...,channel].copy()
        vid_ch = (vid[...,channel]*1.).copy() # ensure this is float. 
        # image_h, image_w = vid_ch.shape[1:]
        
        # iterate over frames. 
        for i in range(len(vid))[:]:
            # frame_file = os.path.join(temp_vid_dir, 'Frame_%s.jpg' %(str(i).zfill(4)))
#                frame_img = vid[i,...,channel].copy()
            frame_img = vid_ch[i].copy()
            
            
            """
            track the original size. 
            """
            image_h_original, image_w_original = frame_img.shape[:2]
            
            
            """
            this should be the original image
            """
            frame_img_viz = frame_img.copy()
            frame_img_viz = np.uint8(255*skexposure.rescale_intensity(frame_img_viz*1.))
            frame_img_viz = np.dstack([frame_img_viz, frame_img_viz, frame_img_viz]) # make RGB. 
            
            """
            resize to run the YOLO network. 
            """
            frame_img = sktform.resize(frame_img, output_shape=imsize, order=1, preserve_range=True) # resize to the size the network expects. 
            
            if equalize_hist:
                frame_img = skexposure.equalize_hist(frame_img) # equalize hist 
            frame_img = np.dstack([frame_img,frame_img,frame_img]) # make RGB.

#            frame_img = np.uint8(255*rgb2gray(vid[i]))
            # frame_img = np.uint8(rescale_intensity(frame_img)) # this seems to work better? 
            # frame_img = np.uint8(255*frame_img)
            frame_img = np.uint8(255*skexposure.rescale_intensity(frame_img))
            # print(frame_img.min(), frame_img.max())
            if (frame_img.min() == frame_img.max()):
                frame_img[:] = 0
                
            image_h, image_w = frame_img.shape[:2]
            # imsave(frame_file, frame_img)
            # imglist.append(frame_file)
            # frame_img_ = preprocess_input(frame_img, net_h, net_w)
            # print('max intensity in, ', np.max(frame_img))
            yolos = yolo_model.predict((frame_img/255.)[None,...]) # outputs 3 heads... for the 3 feature maps...  # this needs to be converted to bbox. ! 
            # yolos = yolo_model.predict(frame_img_)
            boxes = []

            for iii in range(len(yolos)):
                # decode the output of the network
                boxes += _decode_netout(yolos[iii][0], anchors[iii], obj_thresh, nms_thresh, net_h, net_w)

            # correct the sizes of the bounding boxes
            _correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
        
            # suppress non-maximal boxes
            _do_nms(boxes, nms_thresh)     
            
            detect_bboxes = _get_bboxes(boxes, class_labels, obj_thresh=obj_thresh)
            
            
            # get the correct scaling. 
            ### the detection is now correct. # can we correct for the rescaling. 
            scale_x = image_w_original / image_w
            scale_y = image_h_original / image_h
            
            
            # apply the scaling. 
            detect_bboxes_new = []
            for det in detect_bboxes:
                last = np.hstack(det[-1])
                
                last[0] *= scale_x
                last[1] *= scale_y
                last[2] *= scale_x
                last[3] *= scale_y
                
                det[-1] = list(last) 
                
                detect_bboxes_new.append(det)
            
            bbox_img = _draw_bbox(frame_img_viz, detect_bboxes_new, class_labels, cmap) # draw the 
            
            # # draw bounding boxes on the image using labels
            # draw_boxes(frame_img, boxes, class_labels, obj_thresh=0.01)  # ok we get bounding boxes out..... though this will be slightly different ... o boy ... 
            if debug_viz:
                import pylab as plt 
                plt.figure()
                plt.imshow(bbox_img)
                plt.show()

            skio.imsave(os.path.join(pred_folder, 'bbox_frame_%s_Ch-%s.jpg' %(str(i).zfill(4), str(channel+1).zfill(2))), bbox_img)
                
            """
            Write out the bounding box files for each time point in yolo format. 
            """
            outboxfile = os.path.join(pred_folder, 'bbox_predictions_%s_Ch-%s.txt' %(str(i).zfill(4), str(channel+1).zfill(2)))
            write_bbox_files(outboxfile, detect_bboxes_new)
    
    return []
