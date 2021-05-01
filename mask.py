import cv2
import numpy as np
import os
#from numba import njit, cuda
#from numpy import expand_dims
import time

def typeInput(type="video"):
    if type == "video":
        cap = cv2.VideoCapture("video_mask.mp4")
    if type == "cam":
        cap = cv2.VideoCapture(0)
    return cap

def processing_output(img, layerOutputs, width, height, font, classes, count, path):
    boxes, confidences, class_ids = [], [], []
    #print(len(layerOutputs))
    for output in layerOutputs:
        #print(len(output))
        for detection in output:
            #print(detection)
            scores = detection[5:]
            #print(type(scores))
            class_id = np.argmax(scores)
            #print(type(class_id))
            confidence = scores[class_id]
            if confidence > 0.5:
                #print((detection[0:4]))
                box = detection[0:4]*np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                #print(boxes)
                boxes.append([x, y, int(w), int(h)])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2) * 100)
            text = "{} {}% ".format(label, confidence)
            bbox_thick = int(0.6 * (height + width) / 100)
            t_size = cv2.getTextSize(text, 0, 0.5, thickness=bbox_thick // 2)[0]
            print(path+label)
            newtime = time.time()
            newtime = str(newtime)
            count += 1
            count = str(count)
            crop_img = img[y-10:y+h+5, x-10:x+w+10]
            cv2.imwrite(os.path.join(path+label, label+count+newtime+'.jpg'), crop_img)
            if label == 'Wearing Mask':
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img, (x,y), (x + t_size[0]+4, y - t_size[1] - 5), (0,255,0), -1) #filled
                cv2.putText(img, text, (x, y - 2), font, 0.5, (0, 0, 0), 1)
            elif label == 'No Mask':
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x + t_size[0]+4, y - t_size[1] - 5), (0, 0, 255), -1)  # filled
                cv2.putText(img, text, (x, y - 2), font, 0.5, (0, 0, 0), 1)
                #cv2.putText(img, "Put your MASK on now!", (x, y + h + 15), font, 1, (0, 0, 255), 2)


def main():
    path = "/home/rootoversesa/yolo/"
    path_cfg = "models/yolov3-maskk.cfg"
    path_weights = "models/yolov3-maskk_best.weights"
    net = cv2.dnn.readNet(path_weights, path_cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    input_net = 256
    classes = []
    count = 0
    with open("./models/mask.names", "r", encoding='utf_8') as f:
        classes = f.read().splitlines()
    cap = typeInput("cam")
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        _, img = cap.read()
        height, width, _ = img.shape
        t1 = time.time()
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(img, 1/255, (input_net,input_net), (0,0,0), swapRB=True, crop=False)
        print(blob)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        #print("YOLO speed: {:.5f} seconds".format(end-start))
        t = time.time()
        processing_output(img, layerOutputs, width, height, font, classes, count, path)
        mt = time.time() - t
        print("Speed: ", mt)
        fps = 1. / (time.time() - t1)
        #print("Processing Time: ",time.time() - t1)
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), 0, 1, (0, 0, 255), 1)
        cv2.imshow('Mask Detector', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    #box = None
    main()
