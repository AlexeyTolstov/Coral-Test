import cv2, time

from PIL import Image, ImageDraw
import numpy as np

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

print("--- Libraries loaded successfully ---")

def draw_objects(draw, objs, labels):
    for obj in objs:
        bbox = obj.bbox
        
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')

model_path = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
label_path = "coco_labels.txt"
threshold = 0.4
count = 5

labels = read_label_file(label_path) if label_path else {}
print("--- Loading Model ---")
interpreter = make_interpreter(model_path)
print("--- Model loaded successfully ---")
interpreter.allocate_tensors()

cap = cv2.VideoCapture(0)
print("--- Read video from webcamera ---")
while True:
    ret, frame = cap.read()
    st_time = time.time()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    _, scale = common.set_resized_input(
    interpreter, image.size, lambda size: image.resize(size, Image.Resampling.LANCZOS))

    for _ in range(count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(interpreter, threshold, scale)
        print('%.2f ms' % (inference_time * 1000))
        
        draw_objects(ImageDraw.Draw(image), objs, labels)
        image = image.convert('RGB')

    open_cv_image = np.array(image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    
    fps = 1 / (time.time() - st_time)
    cv2.putText(open_cv_image,
                    f"FPS: {fps:.2f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    5)
    
    cv2.imshow('video feed', open_cv_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break