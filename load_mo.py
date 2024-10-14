import cv2
import numpy as np
from tensorflow.keras.models import load_model 

model = load_model("emptyparkingspotdetectionmodel.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

coordinates = [  # Your existing coordinates list
    [(8, 0),(27, 40)],
[(29, 2) , (45, 41)],
[(45, 2),(63, 40)],
[(62, 1),(79, 38)],
[(79, 4),(98, 36)],
[(97, 2) , (118, 34)],
[(119, 0) , (135, 36)],
[(135, 1) , (155, 39)],
[(156, 2), (171, 39)],
[(173, 2) , (191, 36)],
[(191, 2),(204, 36)],
[(209, 6),(225, 39)],
[(227, 3) , (253, 40)],
[(11, 82) , (30, 120)],
[(34, 82) , (47, 119)],
[(49, 82) , (63, 120)],
[(63, 82) , (81, 118)],
[(83, 82) , (98, 115)],
[(100, 83), (117, 116)],
[(116, 83) ,(137, 118)],
[(139, 82), (153, 118)],
[(154, 85) , (171, 117)],
[(173, 81) , (190, 117)],
[(193, 84) , (209, 120)],
[(213, 86) , (224, 120)],
[(226, 82) , (252, 119)],
[(12, 121) , (29, 154)],
[(30, 120) , (45, 157)],
[(49, 119) , (64, 155)],
[(66, 123) , (84, 156)],
[(85, 120) , (100, 156)],
[(101, 122) , (117, 158)],
[(120, 120),(135, 158)],
[(138, 125), (153, 155)],
[(154, 114) , (155, 118)],
[(155, 118),(173, 155)],
[(170, 119), (191, 155)],
[(193, 124),(213, 155)],
[(215, 124), (233, 156)],
[(238, 121), (258, 157)]
]

def detect_empty_parking(image, spot):
    x1, y1 = spot[0]
    x2, y2 = spot[1]
    
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print(f"Invalid coordinates for ROI: {spot}")
        return False

    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        print("Empty ROI")
        return False
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized_roi = cv2.resize(gray_roi, (48,48))
    resized_roi = resized_roi.astype('float32') / 255.0
    resized_roi = np.expand_dims(resized_roi, axis=0)
    resized_roi = np.expand_dims(resized_roi, axis=-1)
    
    prediction = model.predict(resized_roi)
    threshold = 0.5  # Adjusted threshold for more realistic binary classification
    
    return prediction[0][0] > threshold

# Correct the image path and ensure the extension is included
image_path = r"C:\\Users\\Shiva Prasad\\OneDrive\\Documents\\Mini_Proj\\Sample_img\\sa1.jpg"
current_image = cv2.imread(image_path)

# Check if image is loaded successfully
if current_image is None:
    print(f"Error: Could not load image from path {image_path}. Check the file path and try again.")
else:
    empty_count = 0

    for spot in coordinates:
        if detect_empty_parking(current_image, spot):
            cv2.rectangle(current_image, spot[0], spot[1], (0,255,0), 2)  # Green for empty spots
            empty_count += 1
        else: 
            cv2.rectangle(current_image, spot[0], spot[1], (0,0,255), 2)  # Red for occupied spots

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(current_image, f"Empty Spots: {empty_count}", (65,65), font, 0.69, (0,0,0), 2, cv2.LINE_AA)

    desired_width, desired_height = 800, 600
    image_resized = cv2.resize(current_image, (desired_width, desired_height))

    # Display the image
    cv2.imshow('Parking Lot Detection', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
