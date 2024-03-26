import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


class ObjectDetector:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.graph_name = 'detect.tflite'
        self.labelmap_name = 'labelmap.txt'
        self.min_conf_threshold = 0.5
        self.load_model()

    def load_model(self):
        path_to_ckpt = os.path.join(self.model_dir, self.graph_name)
        self.interpreter = Interpreter(model_path=path_to_ckpt)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        with open(os.path.join(self.model_dir, self.labelmap_name), 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def detect_objects(self, image_path, image_name):
        image_path_file = os.path.join(image_path, image_name)
        image = cv2.imread(image_path_file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)

        floating_model = (self.input_details[0]['dtype'] == np.float32)
        if floating_model:
            input_mean = 127.5
            input_std = 127.5
            input_data = (np.float32(input_data) - input_mean) / input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        outname = self.output_details[0]['name']
        if 'StatefulPartitionedCall' in outname:
            boxes_idx, classes_idx, scores_idx = 1, 3, 0
        else:
            boxes_idx, classes_idx, scores_idx = 0, 1, 2

        boxes = self.interpreter.get_tensor(self.output_details[boxes_idx]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[classes_idx]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[scores_idx]['index'])[0]

        detections = []
        for i in range(len(scores)):
            if (scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0):
                detections.append([self.labels[int(classes[i])], f"{int(scores[i] * 100)}%"])

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = self.labels[int(classes[i])]  # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                              cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2)  # Draw label text

        # Save image
        # image_savepath = os.path.join(image_path, image_name)
        cv2.imwrite(image_name, image)

        return detections
