import cv2 as cv
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
from IPython import display

if __name__ == '__main__':
    print(torch.version.cuda)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        device = torch.device('cuda:0')
    else:
        print("Using CPU")

    video_capture = cv.VideoCapture(0)
    if not video_capture.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break

        frames_tracked = []
        for i, frame in enumerate(frame):
            print('\rTracking frame: {}'.format(i + 1), end='')

            # Detect faces
            boxes = MTCNN.detect(frame)

            # Draw faces
            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

            # Add to frame list
            frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
            d = display.display(frames_tracked[0], display_id=True)
            i = 1
            try:
                while True:
                    d.update(frames_tracked[i % len(frames_tracked)])
                    i += 1
            except KeyboardInterrupt:
                pass
        print('\nDone')



    # When everything done, release the capture
    video_capture.release()
    cv.destroyAllWindows()
