import cv2
import random
import pickle
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime

attendance_csv_path = '/home/vaibhavi/Desktop/Facial_recog_project/Attendance.csv'

def attendance(name, attendance_csv_path):
	df = pd.read_csv(attendance_csv_path)
	now = datetime.now()
	time = now.strftime("%H:%M:%S")
	date = now.strftime("%m/%d/%Y")

	# if Attendance.csv is empty or we are storing the first entry of a new day
	if len(df)==0 or df.Date.iloc[-1]!=date:
		with open('Attendance.csv', 'a') as f:
			f.writelines(f'\n{time},{name},{date}')

	# else we will make sure that the person isn't showing up again on the same day, if they are not; we will mark their attendance
	else:
		rslt_df = df.loc[df['Date']==date]
		if name not in list(rslt_df['Name']):
			with open('Attendance.csv', 'a') as f:
				f.writelines(f'\n{time},{name},{date}') 

# loading member names to be shown on screen once identified
with open('names.pkl', 'rb') as f:
	names = pickle.load(f)

# loading the face encodings of members to calculate similarity
with open('saved_encodings.pkl', 'rb') as f:
	encodings = pickle.load(f)

# to make different coloured bounding boxes around detected faces
colours = [(51,51,255), (51, 153, 255), (0, 204, 204), (0,255,128), (0,255,0), (255,255,0), (255,128,0), (255,0,0), (255,0,127), (255,0,255), (127,0,255)]

# for inferencing through web cam
cap = cv2.VideoCapture(0)
while True:
	_, frame = cap.read()
	dim_h = int(frame.shape[0]*(25/100))
	dim_w = int(frame.shape[1]*(25/100))

	small_frame = cv2.resize(frame.copy(), (dim_w, dim_h), interpolation = cv2.INTER_AREA)
	small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

	# location of detected face(s)
	face_locs = face_recognition.face_locations(small_frame)

	# calculating encoding of detected face(s)
	face_encodes = face_recognition.face_encodings(small_frame, face_locs)

	# no face encoding implies no face was detected
	if len(face_encodes)==0:
		cv2.putText(frame, 'NO PERSON FOUND', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, random.choice(colours),3)

	# more than 1 face encoding implies multiple persons trying to mark attendance at the same time; in this case,
	# attendance won't be marked
	elif len(face_encodes)>1:
		cv2.putText(frame, 'ONE PERSON AT A TIME!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, random.choice(colours),3)
		
	# in case of 1 face encoding, we will check if the member is a known person
	else:

		# calculating distance of the obtained face encoding with the face encodings of the members
		facedis = face_recognition.face_distance(encodings, face_encodes[0])

		# if minimum distance is less than 0.5 (at max being 1), then the person is unknown and hence his/her attendance won't be marked
		if min(facedis)>0.5:
			cv2.putText(frame, 'PERSON UNIDENTIFIED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, random.choice(colours),3)
		
		# Amongst the stored face encodings of members, whichever gives minimum value of distance with the obtained face encoding would 
		# be the required person
		else:
			matchidx = np.argmin(facedis)
			name = names[matchidx].upper()
			cv2.putText(frame, f'WELCOME {name}!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, random.choice(colours),3)

			# mark attendance of the identified person
			attendance(name,attendance_csv_path)

	cv2.imshow('frame', frame)
	
	# Press 'Q' to break
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
			
	


