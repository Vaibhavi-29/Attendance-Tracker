import os
import cv2
import pickle
import pandas as pd
import face_recognition

# path to folder having images of people to recognize
path = '/home/vaibhavi/Desktop/Facial_recog_project/members'

# note: images in the above folder should be renamed after the respective member's actual name
encodings = []
members_list = os.listdir(path)

for i, img in enumerate(members_list):
	teammate =  cv2.imread(f'{path}/{img}')
	teammate = cv2.cvtColor(teammate, cv2.COLOR_BGR2RGB)
	teammate_encoding = face_recognition.face_encodings(teammate)[0]
	encodings.append(teammate_encoding)

	# removing the .jpg extension to get the member's actual name
	members_list[i]= img[:-4]
	
print('Encodings calculated')

# ##########	MAKING A CSV FILE FOR STORING ATTENDANCE	##########
# # with open('Attendance.csv', 'w') as f:
# # 	pass

##########	STORING NAMES AND ENCODINGS IN PICKLE FILE	##########
with open('names.pkl', 'wb') as f:
	pickle.dump(members_list, f)

with open('saved_encodings.pkl', 'wb') as f:
	pickle.dump(encodings, f)

with open('names.pkl', 'rb') as f:
	members_list = pickle.load(f)

with open('saved_encodings.pkl', 'rb') as f:
	encodings = pickle.load(f)

print(members_list)
print(encodings)

