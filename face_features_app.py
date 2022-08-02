import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import av
import tensorflow as tf
import os


# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

#             GENDER FILES               
# The gender model architecture
GENDER_PROTO = os.getcwd()+'/asset/gender_files/deploy_gender.prototxt'

# The gender model pre-trained weights
GENDER_MODEL = os.getcwd()+'/asset/gender_files/gender_net.caffemodel'

# Represent the gender classes
GENDER_LIST = ['Male', 'Female']

# Load gender prediction model
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)


#             AGE FILES 
# The age model architecture
AGE_PROTO = os.getcwd()+'/asset/age_files/deploy_age.prototxt'

# The model pre-trained weights
AGE_MODEL = os.getcwd()+'/asset/age_files/age_net.caffemodel'

AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# Load age prediction model
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)


#             SKIN FEATURES FILES
model = tf.keras.models.load_model(os.getcwd()+"/asset/skin_files/Skin-Type-Recognition")
skin_type = ["Dry Skin", "Oily Skin"]

#             FACE DETECTION FILES
face_cascade = cv2.CascadeClassifier(os.getcwd()+'/asset/haarcascades/haarcascade_frontalface_default.xml')

# returns an image with the face bounded and the bounding box indeces
def detect_face(img):
    
    face_bound = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_bound = np.array(face_bound, dtype='uint8')
    
    # perform adaptive equalization for image contrast correction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    face_bound = clahe.apply(face_bound)
    
    # get the points that surrounds the face
    bounds = face_cascade.detectMultiScale(face_bound)
    
    # if there are multiple faces, get the main face (biggest)
    main_bound = [w*h for x,y,w,h in bounds]
    
    # if there's no face detected
    try:
        main_bound = np.argmax(main_bound)
        x,y,w,h = bounds[main_bound]
    except:
        return [0,0,0,0]
    
    return bounds[main_bound]


# returns the proper font size to match the image and face size
def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        
        if (new_width <= width):
            return scale/10
    return 1


# draw the bounding box around the face along with the gender and age
def draw_box(frame, points, gender, gender_confidence_score, age, age_confidence_score, color=(0,0,255)):
    copy = frame.copy()

    x,y,w,h = points

    # draw bounding box
    cv2.rectangle(copy, (x,y), (x+w, y+h), color, 5)

    yPos = y - 15

    # Determine the position of text in y-axis
    while yPos < 15:
        yPos += 15

    # gender label for the box
    label = "Age:{}-{:.2f}%%   {}-{:.2f}%".format(age, age_confidence_score*100, gender, gender_confidence_score*100)
    # print(label)

    # get the font scale for this image size
    optimal_font_scale = get_optimal_font_scale(label, (((x+w)-x)+25))

    # Label processed image
    cv2.putText(copy, label, (x, yPos), cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, color, 2)

    return copy


# returns the gender and its confidence percent for the face showed in the image
def predict_gender(img, points):

    # Get the points of the face
    x,y,w,h = points

    face_img = img[y:y+h, x:x+w]

    # check if face is presence
    try:
        blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
            227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence_score = gender_preds[0][i]

    except:
        return '', 0

    return gender, gender_confidence_score 


#  returns the age and its confidence percent for the face showed in the image
def predict_age(img, points):
    # Get the points of the face
    x,y,w,h = points

    face_img = img[y:y+h, x:x+w]
    
    # check if face is presence
    try: 
        # image --> Input image to preprocess before passing it through our dnn for classification.
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(227, 227), 
            mean=MODEL_MEAN_VALUES, swapRB=False
        )
        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()

        
        i = age_preds[0].argmax()
        age = AGE_INTERVALS[i]
        age_confidence_score = age_preds[0][i]
    except:
        return '', 0

    return age, age_confidence_score


# checks whether the person has an oily or dry skin
def predict_skin(img):
    img  = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, (224,224))

    skin_prob = model.predict(tf.expand_dims(img, axis=0))
    skin = skin_type[skin_prob.argmax()]

    return skin, skin_prob.max()

# image functunality
def get_img():
    image_file = st.file_uploader("Upload your image below:", type=['jpg', 'png', 'jpeg'])

    # no file is detected, program ends
    if not image_file:
        return None

    upload_img = Image.open(image_file)

    # convert to numpy array to perform functions
    frame = np.array(upload_img)

    # Get the information of the face
    points = detect_face(frame)
    gender, gender_confidence_score  = predict_gender(frame, points)
    age, age_confidence_score = predict_age(frame, points)
    skin, skin_prob = predict_skin(upload_img)

    frame = draw_box(frame, points, gender, gender_confidence_score, age, age_confidence_score, color=(0,0,255))

    st.markdown("***")
    st.text('Output:')
    st.image(frame)
    st.text(f'Gender: {gender}-{round(gender_confidence_score*100, 2)}% \nAge:{age}-{round(age_confidence_score*100, 2)}% \nSkin Type: {skin}-{round(skin_prob*100, 2)}%')


# Camera functionality
def get_cam(frame):
    frame = frame.to_ndarray(format='rgb24')

    # Get the information of the face
    points = detect_face(frame)
    gender, gender_confidence_score  = predict_gender(frame, points)
    age, age_confidence_score = predict_age(frame, points)

    frame = draw_box(frame, points, gender, gender_confidence_score, age, age_confidence_score, color=(0,0,255))

    return av.VideoFrame.from_ndarray(frame, format='rgb24')


def main():
    # Arranges the radio buttons horizontally
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    st.title("Facial Feature Detection")
    displayType = st.radio(label = 'Detection method:', options = ['image', 'camera'])

    if displayType == 'image':
        try:
            get_img()
            
        except:
            st.warning('No image was given')

    elif displayType == 'camera':
        # a TURN server to run our camera in-case the user has a slow internet
        # RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        webrtc_streamer(key="example", video_frame_callback=get_cam,  
            rtc_configuration={ 
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            # we only need the camera not the mic
            media_stream_constraints={
                "video": True,
                "audio": False
            }
        )


if __name__ == '__main__':
    main()
