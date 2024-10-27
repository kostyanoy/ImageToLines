import cv2
import numpy as np

from image_to_lines import ImageToLines

video_path = "images/bad_apple.mp4"
model = "models/v2_dense_dropout_88.h5"
result = "result/bad_apple/bad_apple.avi"

conf_m = ImageToLines(confidence=True)
video = cv2.VideoCapture(video_path)

height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(video.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(result, fourcc, fps, (width, height), isColor=True)

frame_num = 0
while(True):
    frame_num += 1
    print(f"{frame_num}/6570")
    ret, frame = video.read()
    res_frame = conf_m.process_image(frame, model)
    res_rgb = np.dstack([np.array(res_frame, dtype=np.uint8)]*3)
    out.write(res_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        break

out.release()
video.release()
cv2.destroyAllWindows()