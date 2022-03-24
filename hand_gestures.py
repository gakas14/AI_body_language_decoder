import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import streamlit as st
import time



# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Mediapipe Solution for hands
mp_hands = mp.solutions.hands



def hand_gesture_decoder(model):
    cap = cv2.VideoCapture(0)

    #cap.set(3, 640)
    #cap.set(4, 480)

    st_frame = st.empty()

    # Initiate Hand model
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            # Color the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            # Recolor the image to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Draw hands landmarks
                # print(results.multi_hand_landmarks)
                if results.multi_hand_landmarks:
                    for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())

                        # print(hand_landmarks)

                        # Extracting hands landmarks
                        hand = hand_landmarks.landmark
                        # print(len(hand))
                        hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten())
                        #print(hand_row)

                    # Make Detection
                    X = pd.DataFrame([hand_row])
                    number_language_class = model.predict(X)[0]
                    number_language_prob = model.predict_proba(X)[0]
                    #print(number_language_class, number_language_prob)

                    # Grab wrist coords
                    coords = tuple(np.multiply(
                        np.array(
                            (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)),
                        [640, 480]).astype(int))

                    cv2.rectangle(image,
                                (coords[0], coords[1] + 5),
                                (coords[0] + len(number_language_class) * 20, coords[1] - 30),
                                (245, 117, 16), -1)
                    cv2.putText(image, number_language_class, coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get the status box
                    cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                    max_prob = round(number_language_prob[np.argmax(number_language_prob)], 2)
                    if max_prob > 0.6:
                        # Display class
                        cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, number_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                        # Display probability

                        cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(max_prob),
                                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass


            #open_img = cv2.imshow('Raw webcam Feed', image)

            st_frame.image(image, channels='BGR', use_column_width=True)
            time.sleep(0)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    st.cap.release()
    cv2.destroyAllWindows()



