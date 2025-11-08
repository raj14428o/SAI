import cv2
import os

frozen_model = r"D:\SAI\obstacle_detection\frozen_inference_graph.pb"
config_file  = r"D:\SAI\obstacle_detection\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
labels_file  = r"D:\SAI\obstacle_detection\coco.names"  

with open(labels_file, 'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')

model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


cap = cv2.VideoCapture(0)  # 0 = default webcam (replace with video path if needed)

if not cap.isOpened():
    print(" Error: Cannot open camera.")
    exit()

# Read first frame to init writer
ret, frame = cap.read()
if not ret:
    print(" Error: Cannot read from camera.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output_video.avi', fourcc, 25, (frame.shape[1], frame.shape[0]))

font = cv2.FONT_HERSHEY_PLAIN
print(" Model and video stream initialized successfully!")


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not captured. Exiting loop...")
            break

        # Detect objects
        classIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

        if len(classIndex) != 0:
            for classInd, conf, box in zip(classIndex.flatten(), confidence.flatten(), bbox):
              
                if 0 < classInd <= len(classLabels):
                    label = f"{classLabels[classInd - 1].upper()} {round(conf * 100, 1)}%"
                else:
                    label = f"UNKNOWN {round(conf * 100, 1)}%"

                cv2.rectangle(frame, box, (255, 0, 0), 2)
                cv2.putText(frame, label, (box[0] + 10, box[1] + 30),
                            font, 1, (0, 255, 0), 2)

        # Save and display
        video.write(frame)
        cv2.imshow('Live Object Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Exiting detection loop...")
            break


except Exception as e:
    print(" Exception occurred:", e)

finally:
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    print("Resources released successfully!")




