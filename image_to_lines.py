import math

import cv2
import keras
import numpy as np


# main class to manage model results
class ImageToLines:
    def __init__(self, h=5, w=5, confidence=False):
        self.h = h  # window height
        self.w = w  # window width
        self.confidence = confidence  # large lines mode
        self.lines = {
            0: lambda im, x, y, t, c, conf: 1,  # cv2.line(im, (x, y), (x, y), c, t),
            1: lambda im, x, y, t, c, conf: cv2.line(im, (round(x - self.h * 0.5 * conf), y),
                                                     (round(x + self.w * 0.5 * conf), y), c, t),
            2: lambda im, x, y, t, c, conf: cv2.line(im, (x, round(y - self.h * 0.5 * conf)),
                                                     (x, round(y + self.h * 0.5 * conf)), c, t),
            3: lambda im, x, y, t, c, conf: cv2.line(im,
                                                     (round(x - self.w * 0.5 * conf), round(y + self.h * 0.5 * conf)),
                                                     (round(x + self.w * 0.5 * conf), round(y - self.h * 0.5 * conf)),
                                                     c,
                                                     t),
            4: lambda im, x, y, t, c, conf: cv2.line(im,
                                                     (round(x + self.w * 0.5 * conf), round(y - self.h * 0.5 * conf)),
                                                     (round(x - self.w * 0.5 * conf), round(y + self.h * 0.5 * conf)),
                                                     c, t)
        }  # represent model results in lines

    # draw line on image
    def draw_line(self, im, x, y, prediction):
        confidence = np.max(prediction) if self.confidence else 1
        label = np.argmax(prediction)
        color = (0, 0, 0)
        thickness = 1

        operation = self.lines[label]
        operation(im, x, y, thickness, color, confidence)

    # give image windows to model and draw lines
    def process_image(self, im_path, mod_path):
        image = cv2.imread(im_path)  # original image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale image
        h, w = gray_image.shape

        empty_image = np.zeros((h, w)) + 255  # white image

        s_i = self.h // 2
        e_i = h - self.h // 2
        s_j = self.w // 2
        e_j = w - self.w // 2

        model = keras.models.load_model(mod_path)

        areas = []
        for i in range(s_i, e_i, self.h):
            for j in range(s_j, e_j, self.w):
                area = np.array(
                    gray_image[i - self.h // 2:i + math.ceil(self.h / 2), j - self.w // 2:j + math.ceil(self.w / 2)]
                )
                areas.append(area)

        areas = np.array(areas)  # windows from image
        p = model.predict(areas, verbose=2)  # model results for windows
        c = 0

        for i in range(s_i, e_i, self.h):
            for j in range(s_j, e_j, self.w):
                self.draw_line(empty_image, j, i, p[c])
                c += 1

        return empty_image
