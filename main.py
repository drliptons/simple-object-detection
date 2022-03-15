from detector import Detector


class_file = 'coco.names'
model_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz'
# model_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz'

image_list = ['test/car.jpg', 'test/japan.jpg', 'test/saint-peters.jpg', 'test/times-square.jpg', 'test/usa.jpg']
image = 'test/car.jpg'
video = 'test/chelsea.mp4'  # for video clip
# video = 0  # for webcam
confidence = 0.5
overlap = 0.5

detector = Detector()
detector.read_class(class_file_path=class_file)
detector.download_model(model_url)
detector.load_model()
# Predict single image
# detector.predict_image(image, confidence=confidence, overlap=overlap)
# Predict multiple images
for image in image_list:
    detector.predict_image(image, confidence=confidence, overlap=overlap)
# Predict video
# detector.predict_video(video, confidence=confidence, overlap=overlap)
