import pickle
from hand_gestures import hand_gesture_decoder
import streamlit as st
from body_language import body_language_decoder


# import the model
with open('hand_gestures.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Webcam Application')

selected_model = st.sidebar.selectbox('Pick the model', ['Body language Decoder', 'Hand Gestures Decoder'])


if selected_model == 'Hand Gestures Decoder':
    with open('hand_gestures.pkl', 'rb') as f:
        model = pickle.load(f)
    hand_gesture_decoder(model)
else:
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    body_language_decoder(model)





# # Drawing helpers
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# # Mediapipe Solution for hands
# mp_hands = mp.solutions.hands
#
# cap = cv2.VideoCapture(0)
# st_frame = st.empty()
# ret, img = cap.read()
# # Initiate Hand model
# with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     #while ret:
#     while cap.isOpened():
#         #ret, img = cap.read()
#         success, image = cap.read()
#         #st_frame.image(image, channels='BGR')
#
#         image.flags.writeable = False
#         # Color the image to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = hands.process(image)
#
#         # Draw the hand annotations on the image.
#         image.flags.writeable = True
#         # Recolor the image to BGR
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         #st_frame.image(image)
#         try:
#             # Draw hands landmarks
#             # print(results.multi_hand_landmarks)
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                                               mp_drawing_styles.get_default_hand_landmarks_style(),
#                                               mp_drawing_styles.get_default_hand_connections_style())
#
#                     # print(hand_landmarks)
#
#                     # Extracting hands landmarks
#                     hand = hand_landmarks.landmark
#                     # print(len(hand))
#                     hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten())
#                     # print(hand_row)
#
#                 # Make Detection
#                 X = pd.DataFrame([hand_row])
#                 number_language_class = model.predict(X)[0]
#                 number_language_prob = model.predict_proba(X)[0]
#                 # print(number_language_class, number_language_prob)
#
#                 # Grab wrist coords
#                 coords = tuple(np.multiply(
#                     np.array(
#                         (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
#                          hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)),
#                     [640, 480]).astype(int))
#
#                 cv2.rectangle(image,
#                               (coords[0], coords[1] + 5),
#                               (coords[0] + len(number_language_class) * 20, coords[1] - 30),
#                               (245, 117, 16), -1)
#                 cv2.putText(image, number_language_class, coords,
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#                 # Get the status box
#                 cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
#
#                 # Display class
#                 cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#                 cv2.putText(image, number_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                             (255, 255, 255), 2, cv2.LINE_AA)
#
#                 # Display probability
#                 cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#                 cv2.putText(image, str(round(number_language_prob[np.argmax(number_language_prob)], 2)),
#                             (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#         except:
#             pass
#
#         st_frame.image(image, channels='BGR')
#
#         #if cv2.waitKey(10) & 0xFF == ord('q'):
#             #break
#
