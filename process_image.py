import cv2

from image_to_lines import ImageToLines

image_path = "dog.jpg"  # image name

image_folder = "images"
model_folder = "models"
result_folder = "result/dog"

models = [r"v1_dense86.h5", r"v2_dense_dropout_88.h5", r"v3_conv75.h5"]  # model names

not_conf_m = ImageToLines(confidence=False)
conf_m = ImageToLines(confidence=True)

# run each model and save results
for m in models:
    im1 = not_conf_m.process_image_from_file(f"{image_folder}/{image_path}", f"{model_folder}/{m}")
    im2 = conf_m.process_image_from_file(f"{image_folder}/{image_path}", f"{model_folder}/{m}")

    cv2.imwrite(f"{result_folder}/confident_{m}.jpg", im1)
    cv2.imwrite(f"{result_folder}/not_confident_{m}.jpg", im2)
