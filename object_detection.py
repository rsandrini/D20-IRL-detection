import os
import cv2
import numpy as np
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter


class ObjectDetector:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.min_conf_threshold = 0.3
        self.nms_threshold = 0.45
        self.labels = [str(i) for i in range(1, 21)]
        self.load_model()

    def load_model(self):
        model_path = os.path.join(self.model_dir, "detect.tflite")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_height = self.input_details[0]["shape"][1]
        self.input_width = self.input_details[0]["shape"][2]
        self.is_quantized = self.input_details[0]["dtype"] == np.uint8

    def _preprocess(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
        input_data = np.expand_dims(resized, axis=0)
        if self.is_quantized:
            return input_data.astype(np.uint8)
        return (input_data.astype(np.float32) / 255.0)

    def detect_objects(self, image_path, image_name):
        image_path_file = os.path.join(image_path, image_name)
        image = cv2.imread(image_path_file)
        imH, imW = image.shape[:2]

        self.interpreter.set_tensor(self.input_details[0]["index"], self._preprocess(image))
        self.interpreter.invoke()

        # YOLOv8 TFLite output: [1, 4+nc, num_anchors] — transpose to [num_anchors, 4+nc]
        output = np.squeeze(self.interpreter.get_tensor(self.output_details[0]["index"])).T
        boxes_raw = output[:, :4]      # cx, cy, w, h in input image pixel coords
        class_scores = output[:, 4:]   # nc scores per anchor

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences >= self.min_conf_threshold
        if not mask.any():
            return []

        boxes_raw = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Scale from model input coords to original image size
        sx = imW / self.input_width
        sy = imH / self.input_height
        xmin = np.clip((boxes_raw[:, 0] - boxes_raw[:, 2] / 2) * sx, 0, imW).astype(int)
        ymin = np.clip((boxes_raw[:, 1] - boxes_raw[:, 3] / 2) * sy, 0, imH).astype(int)
        bw = (boxes_raw[:, 2] * sx).astype(int)
        bh = (boxes_raw[:, 3] * sy).astype(int)

        nms_boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h in zip(xmin, ymin, bw, bh)]
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), self.min_conf_threshold, self.nms_threshold)
        if len(indices) == 0:
            return []

        detections = []
        for i in indices.flatten()[:2]:
            label = self.labels[class_ids[i]]
            detections.append([label, f"{int(confidences[i] * 100)}%"])

            x, y, w, h = nms_boxes[i]
            xmax, ymax = x + w, y + h
            cv2.rectangle(image, (x, y), (xmax, ymax), (10, 255, 0), 2)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ly = max(y, label_size[1] + 10)
            cv2.rectangle(image, (x, ly - label_size[1] - 10), (x + label_size[0], ly + 5), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (x, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imwrite(image_path_file, image)
        return detections
