import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mp_face_detection = mp.solutions.face_detection


def detect_face(img):
    crops, boxes = [], []
    image_rows, image_cols, _ = img.shape
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img)
        
        if not results.detections:
            return crops, boxes
        
        
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
            
            xleft,ytop=rect_start_point
            xright,ybot=rect_end_point

            crops.append(img[ytop: ybot, xleft: xright])
            boxes.append((xleft, ytop, xright, ybot))
        
        return crops, boxes

        
        