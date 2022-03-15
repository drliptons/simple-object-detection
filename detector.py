import os
import time
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file


np.random.seed(42)


class Detector:
    def __init__(self):
        self.class_list = None
        self.color_list = None
        self.model = None
        self.model_name = None

        # Create 'pretrained_models' folder
        self.cache_dir = './pretrained_models'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Create save predicted images folder
        self.path = './predicted_images'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def read_class(self, class_file_path):
        # Open class names file
        with open(class_file_path, 'r') as f:
            # Assign class names file to a variable
            self.class_list = f.read().splitlines()

        # Assign colors to each class
        self.color_list = np.random.uniform(low=0, high=256, size=(len(self.class_list), 3))

    def download_model(self, model_url):
        # Extract model name
        file_name = os.path.basename(model_url)
        self.model_name = file_name[:file_name.index('.')]  # get name

        # Download and extract the model
        get_file(fname=file_name, origin=model_url, cache_dir=self.cache_dir, cache_subdir='model_zoo', extract=True)

    def load_model(self):
        print(f"Loading model {self.model_name}")
        start = time.time()
        # Clear all previous sessions
        tf.keras.backend.clear_session()
        # Load model
        self.model = tf.saved_model.load(os.path.join(self.cache_dir, 'model_zoo', self.model_name, 'saved_model'))
        print(f"Model {self.model_name} loaded success in {round((time.time() - start), 4)} seconds")

    def create_bounding_box(self, image, confidence=0.5, overlap=0.5):
        # Convert input to RGB format
        input_tensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)  # return a np.array
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)  # convert into tensors
        input_tensor = input_tensor[tf.newaxis, ...]  # tf takes batch as input, convert input into batch and expand its dim

        # Pass input into the model and get detections as return
        detections = self.model(input_tensor)

        # Extract all needed features and covert to numpy arrays
        bboxs = detections['detection_boxes'][0].numpy()  # bbox
        class_indexes = detections['detection_classes'][0].numpy().astype(np.int32)  # class indexes
        class_scores = detections['detection_scores'][0].numpy()  # confidence score for each labels

        # Get image height, width, and channel to calculate bbox position
        image_height, image_width, image_channel = image.shape

        # Clean overlapping bboxes
        # max_output_size: number of maximum bboxes
        # iou_threshold: overlapping bbox above this value will be discarded. Calculate by area of overlap / total area
        # score_threshold: minimum confidence score threshold to keep the bbox
        bbox_idx = tf.image.non_max_suppression(bboxs, class_scores, max_output_size=50, iou_threshold=overlap,
                                                score_threshold=confidence)

        if len(bbox_idx) != 0:
            for i in bbox_idx:
                bbox = tuple(bboxs[i].tolist())  # class bbox
                class_confidence = round(class_scores[i] * 100)  # class confidence score and round
                class_index = class_indexes[i]  # class label

                class_label_text = self.class_list[class_index].upper()  # get class name
                class_color = self.color_list[class_index]  # get class color

                display_text = f"{class_label_text}: {class_confidence}%"

                # Get bbox position
                ymin, xmin, ymax, xmax = bbox

                # Calculate bbox dimensions
                xmin, xmax, ymin, ymax = (int(xmin * image_width), int(xmax * image_width),
                                          int(ymin * image_height), int(ymax * image_height))

                # Draw bboxes
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=class_color, thickness=2)  # draw bbox

                # Draw text
                font = cv2.FONT_HERSHEY_PLAIN  # set font
                font_scale = 1.5  # set font scale
                font_thickness = 2  # set font thickness
                font_color = (255, 255, 255)  # set font color
                y_margin = 10  # set text y margin
                (text_width, text_height), _ = cv2.getTextSize(display_text, font, font_scale, font_thickness)
                cv2.rectangle(image, (xmin, ymin - text_height - y_margin * 2), (xmin + text_width, ymin),
                              color=class_color, thickness=-1)  # draw text background
                cv2.putText(image, display_text, (xmin + 1, ymin - y_margin), font, font_scale, font_color,
                            font_thickness)  # draw text

                # (Optional) Draw thick bbox edges
                line_width = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))
                # Corner 1
                cv2.line(image, (xmin, ymin), (xmin + line_width, ymin), class_color, thickness=5)
                cv2.line(image, (xmin, ymin), (xmin, ymin + line_width), class_color, thickness=5)
                # Corner 2
                cv2.line(image, (xmax, ymin), (xmax - line_width, ymin), class_color, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin + line_width), class_color, thickness=5)
                # Corner 3
                cv2.line(image, (xmin, ymax), (xmin + line_width, ymax), class_color, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax - line_width), class_color, thickness=5)
                # Corner 4
                cv2.line(image, (xmax, ymax), (xmax - line_width, ymax), class_color, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax - line_width), class_color, thickness=5)

        # Return finished image
        return image

    def predict_image(self, image_path, confidence=0.5, overlap=0.5):
        start = time.time()
        # Load image
        image = cv2.imread(image_path)

        bbox_image = self.create_bounding_box(image, confidence, overlap)  # store image in local variable

        # Show image
        image_name = os.path.basename(image_path)  # get image name
        save_name = image_name[:image_name.index(".")] + '_' + self.model_name + '.jpg'  # define save name
        print(f'Open {image_name}')
        cv2.imwrite(os.path.join(self.path, save_name), bbox_image)  # save detected image
        cv2.imshow(f'Image Detection - {image_name}', bbox_image)  # show image
        print(f'{image_name} time: {round((time.time() - start), 4)}')
        cv2.waitKey(0)  # continue only when pressing a key
        cv2.destroyAllWindows()  # close all windows

    def predict_video(self, video_path, confidence=0.5, overlap=0.5):
        # Open video file
        capture = cv2.VideoCapture(video_path)

        # Check if video is open
        if not capture.isOpened():
            print('Error while opening the video file')
            return

        # Capturing image
        (success, image) = capture.read()

        start_time = 0
        while success:
            now = time.time()

            # Calculate fps
            fps = 1/(now - start_time)
            start_time = now

            # Draw bbox
            bbox_image = self.create_bounding_box(image, confidence, overlap)
            cv2.putText(bbox_image, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow('Video Detection', bbox_image)

            # Close all by press the key 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            (success, image) = capture.read()

        cv2.destroyAllWindows()
