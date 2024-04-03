import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


class BoxDetection:
    def __init__(self, detection_x, detection_y, detection_width, detection_height, label_x, label_y, label_width,
                 label_height, label_text, label_text_x, label_text_y):
        self.detection_x = detection_x
        self.detection_y = detection_y
        self.detection_width = detection_width
        self.detection_height = detection_height

        self.label_x = label_x
        self.label_y = label_y
        self.label_width = label_width
        self.label_height = label_height

        self.label_text = label_text
        self.label_text_x = label_text_x
        self.label_text_y = label_text_y

    def detection_box(self):
        return (self.detection_x, self.detection_y), (self.detection_width, self.detection_height)

    def label_box(self):
        return (self.label_x, self.label_y), (self.label_width, self.label_height)

    def label_box_width_height(self):
        return (self.label_width, self.label_height)

    def label_text_position(self):
        return (self.label_text_x, self.label_text_y)


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
        # new file name for the result
        image_path_new_file = os.path.join(image_path, f"{image_name.split('.')[0]}_result.jpg")
        image = cv2.imread(image_path_file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        print(f"Image dimensions: {imW}x{imH}")
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
        boxes_data = []

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
                label_ymin = max(label_size[1], ymin)  # Ensure label doesn't extend beyond top of the image
                label_xmin = xmin

                # RIGHT SIDE TOP
                if label_xmin < imW / 2 and label_ymin < imH / 3:
                    label_xmin = xmin - 15

                # LEFT SIDE TOP
                elif label_xmin > imW / 2 and label_ymin < (imH / 3):
                    label_xmin = xmax

                if label_ymin < label_size[1] + 10:
                    label_ymin = ymin + label_size[1] + 10 # Move label above the box if it extends beyond the top

                cv2.rectangle(image,
                              (xmin, ymin),
                              (xmax, ymax),
                              (10, 255, 0),
                              2)

                box = BoxDetection(xmin, ymin, xmax, ymax, label_xmin, label_ymin, label_xmin + label_size[0], label_ymin + 5, label_text, label_xmin, label_ymin)

                boxes_data.append(box)  # Store label and its y-coordinate

        # create a list with all boxes from boxes_date getting  box.detection_box(), box.label_box()
        all_boxes = [[box.detection_box(), box.label_box()] for box in boxes_data]
        # keep it with one list
        all_boxes = [item for sublist in all_boxes for item in sublist]

        print(all_boxes)
        print()

        for i, box_data in enumerate(boxes_data):
            # white_box_start, white_box_end, box_label, detection_box_start, detection_box_end = box_data
            print(f"Checking {box_data.label_box()} collision")

            if len(boxes_data) == 2:  # If only one die was detected, we can skip this step
                # Get the two boxes inside the opoosite object
                if self.is_collision(box_data.label_box(),
                                     [boxes_data[1].detection_box(), boxes_data[1].label_box()] if i == 0
                                     else [boxes_data[0].detection_box(), boxes_data[0].label_box()]):
                    print("Collision detected, adjusting label position")
                    new_x, new_y = self.find_clear_position([imH + 10, imW + 10],
                                                            all_boxes,
                                                            box_data.label_box_width_height())

                    # Adjust the text inside the white box
                    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                    box_data.label_x = new_x
                    box_data.label_y = new_y
                    box_data.label_width = new_x + label_size[0]
                    box_data.label_height = new_y + label_size[1]

            print(f"Drawing white box on: {box_data.label_x} - {box_data.label_y}")
            cv2.rectangle(image,
                          (box_data.label_x, box_data.label_y),
                          (box_data.label_width, box_data.label_height),
                          (255, 255, 255),
                          cv2.FILLED)

            print(f"Text on: ({box_data.label_text_x}, {box_data.label_text_y})")

            cv2.putText(image,
                        box_data.label_text,
                        (box_data.label_text_x, box_data.label_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)

        # Save image in a new file
        cv2.imwrite(image_path_new_file, image)

        return detections, image_path_new_file

    def is_collision(self, rect1, rects):

        (x1, y1), (w1, h1) = rect1
        for (x, y), (w, h) in rects:
            if (x1 < x + w and x1 + w1 > x and
                    y1 < y + h and y1 + h1 > y):
                return True
        return False

    def find_clear_position(self, boundary, rectangles, new_rect_size, step=1):
        """
        Find a clear position for a new rectangle that doesn't collide with any of the existing rectangles.

        :param boundary: A tuple containing the width and height of the search area (w, h).
        :param rectangles: A list of existing rectangles in the format (x, y, w, h).
        :param new_rect_size: A tuple containing the width and height of the new rectangle (w, h).
        :param step: The step size to move in the search area. Default is 1.
        :return: A tuple (x, y) representing the top-left corner of the first clear position found, or None if no clear position is found.
        """
        boundary_w, boundary_h = boundary
        new_w, new_h = new_rect_size

        for y in range(0, boundary_h - new_h + 1, step):
            for x in range(0, boundary_w - new_w + 1, step):
                new_rect = ((x, y), (new_w, new_h))
                collision_found = False
                # from pprint import pprint
                # pprint(rectangles)
                # print()
                for rect_start, rect_end, _, detect_start, detect_end in rectangles:
                    if self.is_collision(new_rect, [(rect_start, rect_end), (detect_start, detect_end)]):
                        collision_found = True
                        break
                if not collision_found:
                    return (x, y)
        return None
