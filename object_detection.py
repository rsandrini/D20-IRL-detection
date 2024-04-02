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
        all_labels = []
        for i in range(len(scores)):
            if (scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0 and len(detections) <= 1):
                detections.append([self.labels[int(classes[i])], f"{int(scores[i] * 100)}%"])

                # Get bounding box coordinates and draw box
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                print(f"Object {i}: {self.labels[int(classes[i])]} ({xmin}, {ymin}) ({xmax}, {ymax})")

                label_text = self.labels[int(classes[i])]
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(label_size[1] + 10, ymin)  # Ensure label doesn't extend beyond top of the image
                label_xmin = xmin

                if label_ymin < label_size[1] + 10:
                    label_ymin = ymin + label_size[1] + 10  # Move label above the box if it extends beyond the top

                # Check for collision with other labels
                for other_label in all_labels:
                    other_label_rect = cv2.getTextSize(other_label[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                    if self.is_collision((xmin, label_ymin, label_size[0], label_size[1]), other_label_rect):
                        # Adjust current label to a clear position
                        label_ymin = max(other_label[1] + label_size[1] + 10, ymin + label_size[1] + 10)  # Ensure label doesn't overlap with other labels
                        label_xmin = xmin - 10


                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                cv2.rectangle(image, (xmin, label_ymin - label_size[1] - 10),
                              (xmin + label_size[0], label_ymin + 5), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label_text, (label_xmin, label_ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                print(f"Text on: ({label_xmin}, {label_ymin})")

                all_labels.append((label_text, label_ymin))  # Store label and its y-coordinate

        # Save image
        cv2.imwrite(image_path_file, image)

        return detections

    def is_collision(self, rect1, rect2):
        # Check for collision between two rectangles
        if len(rect2) < 4:
            return False

        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):
            return True
        return False



