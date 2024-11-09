import torch
import cv2
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load the pre-trained Faster R-CNN model trained on the COCO dataset
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set to evaluation mode

# Define COCO dataset class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'TV', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define a function to process and detect objects in each video frame
def detect_objects_in_frame(frame, threshold=0.8):
    # Convert the frame to PIL format for processing
    transform = T.Compose([T.ToTensor()])
    frame_tensor = transform(frame).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(frame_tensor)

    # Extract bounding boxes, labels, and scores
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Filter out detections below the confidence threshold
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score >= threshold:
            # Convert tensor to integer for drawing on frame
            x1, y1, x2, y2 = map(int, box)
            label_text = f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}"
            
            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Process video frames in real-time
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the current frame
    frame = detect_objects_in_frame(frame)

    # Display the frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
