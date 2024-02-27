'''command to run in cmnd prompt 
    python -m streamlit run final.py
'''
import streamlit as st
import cv2
import av
import numpy as np

# Load the pre-trained model
net = cv2.dnn.readNetFromDarknet('yolov3-obj.cfg', 'yolov3-helmet.weights')

# Define the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]

# Function to perform helmet detection
def detect_helmet(image):
    # Perform forward pass
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Get bounding box coordinates and class predictions
    boxes = []
    confidences = []
    class_ids = []
    (height, width, channels) = image.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Separate helmet and no helmet bounding boxes
    helmet_boxes = []
    no_helmet_boxes = []

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if class_ids[i] == 0:  # Class ID 0 for helmet
                helmet_boxes.append([x, y, w, h])
            else:
                no_helmet_boxes.append([x, y, w, h])

    return image, helmet_boxes, no_helmet_boxes

# Streamlit app

def detect_helmet_video(frame):
        # Perform forward pass
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Get bounding box coordinates and class predictions
        boxes = []
        confidences = []
        class_ids = []
        (height, width) = frame.shape[:2]

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_id == 0:  # Class ID 0 for helmet
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate top-left corner coordinates of the bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression to eliminate redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on the frame
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Helmet', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame


def image_detect():
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read the image file
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Perform helmet detection
        output_image, helmet_boxes, no_helmet_boxes = detect_helmet(image)

        # Draw bounding boxes and labels on the image
        for x, y, w, h in helmet_boxes:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, 'Helmet', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for x, y, w, h in no_helmet_boxes:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(output_image, 'No Helmet', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the input and output images
        st.image([output_image], caption=['Output Image'], width=300)

def video_detect():
# Load the pre-trained model
    net = cv2.dnn.readNetFromDarknet('yolov3-obj.cfg', 'yolov3-custom_7000.weights')

    # Define the output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]

    # Function to perform helmet detection

    uploaded_file = st.file_uploader('Upload a video', type=['mp4'])

    if uploaded_file is not None:
        # Open the video file using av
        container = av.open(uploaded_file)
        placeholder=st.empty()
        out=None
        # Iterate over the frames in the video
        for frame in container.decode(video=0):
            # placeholder.empty()
            # Convert the frame to BGR format
            frame = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)

            # Perform helmet detection
            output_frame = detect_helmet_video(frame)
            placeholder.image(output_frame,channels='BGR')
            # height,width, layers=output_frame.shape
            # size=(height,width)
            # if out is None:
            #     out=cv2.VideoWriter('video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),15,size)
            # out.write(output_frame)

            # Display the output frame
            # placeholder.image(output_frame, channels='BGR', caption='Helmet Detection')
        # video_file = open("video.mp4", 'rb')
        # vb=video_file.read()
        # st.video(vb)
        container.close()


st.title('Helmet Detection')


a=st.selectbox('Select the type of data', ('Image', 'Video'))
if a=='Image':
    image_detect()
else:
    video_detect()