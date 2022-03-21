import torch
import numpy as np
import cv2
import pafy
import pandas as pd

from time import time


class ObjectDetection:

    def __init__(self, url, out_file="Labeled_Video.avi"):
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_video_from_url(self):
        play = pafy.new(self._URL).streams[-1]
        assert play is not None
        return cv2.VideoCapture(play.url)

        #For Webcam
        # return cv2.VideoCapture(0)

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        player = self.get_video_from_url()
        # assert player.isOpened()
        # x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        # y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        # out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))



        # while True:
        #     start_time = time()
        #     ret, frame = player.read()
        #     assert ret
        #     results = self.score_frame(frame)
        #     frame = self.plot_boxes(results, frame)
        #     end_time = time()
        #     fps = 1/np.round(end_time - start_time, 3)
        #     print(f"Frames Per Second : {fps}")
        #     out.write(frame)

        if (player.isOpened() == False):
            print("Error opening video")

        while (player.isOpened()):
            ret, frame = player.read()
            if ret == True:
                start_time = time()
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)
                end_time = time()
                fps = 1 / np.round(end_time - start_time, 3)
                format_fps = "{:.2f}".format(fps)
                print(f"Frames Per Second : {format_fps}")
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f'{format_fps} Frames per Sec', (0, int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))-100), font, 2, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        player.release()
        cv2.destroyAllWindows()

# Create a new object and execute.

inp = input("Enter YT URL : ")
a = ObjectDetection(inp)
a()
