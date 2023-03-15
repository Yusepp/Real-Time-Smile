import mediapipe as mp
import numpy as np  
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from PIL import Image

import matplotlib.pyplot as plt
import io
import cv2

mp_face_detection = mp.solutions.face_detection


def detect_face(img):
    img = np.array(img)
    crops, boxes = [], []
    image_rows, image_cols, _ = img.shape
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img)
        
        if not results.detections:
            return crops, boxes
        
        try:
            for det in results.detections:
                location = det.location_data
                relative_bounding_box = location.relative_bounding_box
                rect_start_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                    image_rows)
                rect_end_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                    image_rows)
                
                xleft, ytop = rect_start_point
                xright, ybot = rect_end_point

                crops.append(Image.fromarray(img[ytop: ybot, xleft: xright]))
                boxes.append((xleft, ytop, xright, ybot))
        except:
            return crops, boxes
        
        return crops, boxes


def get_intensity_plot_array(values):
    y = values
    x = [i for i in range(len(y))]
    
    plt.plot(x, y, color='blue')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Smile Intensity', fontsize=12)
    plt.ylim(0, 1)
    fig = plt.gcf()
    
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', dpi=200)
    io_buf.seek(0)
    flat_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(flat_arr, 1)
    io_buf.close()
    
    return img
    