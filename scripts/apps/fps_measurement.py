import time

import cv2
import numpy as np

import head_segmentation.segmentation_pipeline as seg_pipeline


def main() -> None:
    OPENCV_WINDOW_NAME = "Human Head Segmentation FPS Measurement"
    cv2.namedWindow(OPENCV_WINDOW_NAME, cv2.WINDOW_NORMAL)

    web_camera = cv2.VideoCapture(0)

    prev_frame_time = 0
    next_frame_time = 0

    segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline()

    while web_camera.isOpened():
        ret, frame = web_camera.read()
        if not ret:
            break

        segmap = segmentation_pipeline.predict(frame)

        next_frame_time = time.time()

        fps = round(1 / (next_frame_time - prev_frame_time), 4)
        prev_frame_time = next_frame_time

        display_frame = np.hstack(
            (
                frame,
                cv2.cvtColor(segmap * 255, cv2.COLOR_GRAY2RGB),
                frame * cv2.cvtColor(segmap, cv2.COLOR_GRAY2RGB),
            )
        )
        display_frame = cv2.putText(
            img=display_frame,
            text=f"FPS = {str(fps)}",
            org=(15, 55),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 255, 0),
            thickness=5,
        )

        cv2.imshow(OPENCV_WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    web_camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
