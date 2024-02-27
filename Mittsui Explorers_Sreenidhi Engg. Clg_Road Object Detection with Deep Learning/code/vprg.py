'''
command to run the prog
python vprg.py -v production_id.mp4 -c yolov3.cfg -w yolov3.weights -cl yolov3.txt
'''

import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
#ap.add_argument('-v', '--video', required=True, help='C:/Users/Sudhasree D/Mynotebook/intel_unnati/road_trafifc.mp4')
ap.add_argument('-v', '--video', required=True, help='C:/Users/user/Desktop/Mittsui Explorers_Sreenidhi Engg. Clg_Road Object Detection with Deep Learning/data/road_traffic.mp4')
ap.add_argument('-c', '--config', required=True, help='C:/Users/user/Desktop/Mittsui Explorers_Sreenidhi Engg. Clg_Road Object Detection with Deep Learning/code/yolov3.cfg')
ap.add_argument('-w', '--weights', required=True, help='C:/Users/user/Desktop/Mittsui Explorers_Sreenidhi Engg. Clg_Road Object Detection with Deep Learning/code/yolov3.weights')
ap.add_argument('-cl', '--classes', required=True, help='C:/Users/user/Desktop/Mittsui Explorers_Sreenidhi Engg. Clg_Road Object Detection with Deep Learning/code/yolov3.txt')

args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    # Exclude classes you want to remove
    classes_to_exclude = ["backpack", "boat", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball","kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle","wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant","bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush", "bench"]

    if label not in classes_to_exclude:
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 4)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

cap = cv2.VideoCapture(args.video)

#nbtest to store the video 
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

#
while True:
    ret, frame = cap.read()

    if not ret:
        break

    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                #nb test
                #print ("x y ",x,y,w,h)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        #nbtest
        imageCopy = frame.copy()
        cv2.imwrite('Frame'+str(i)+'.jpeg', imageCopy)
        #cv2.imshow ("image copy",imageCopy)
        #nbtest
        writer.write(frame)
        cv2.imshow("object detection", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
