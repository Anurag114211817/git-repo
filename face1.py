# imports --------------------------------
import face_recognition
from datetime import datetime
from pandas import DataFrame, read_csv
import cv2
from os import listdir, chdir, system, remove
import numpy as np
from gtts import gTTS

# change directory -----------------------
chdir('Data')

# global variables -----------------------
audio_file = 'Audio.wav'
filename = 'Data_file.csv'
video_capture = cv2.VideoCapture(0)
face_locations = []
face_encodings = []
face_names = []
name = 'Unknown'
face = ''
process_this_frame = True
df = read_csv(filename)
known_face_names = list(df['Name'])
known_face_encodings = [np.array(list(map(float, list(df['Data'])[i].replace('[', '').replace(']', '').replace('\n', '').split()))) for i, j in enumerate(known_face_names)]

# change directory -----------------------
chdir('../Images')

# load img folder data -------------------
img_folder_data = [x.split('.')[0] for x in listdir()]

if (new_data := set(img_folder_data).difference(set(known_face_names))) != set():
	
	# printing new data found ----------------
	print(f'New data found!\nData is updating\n{list(new_data)}')
	
	# local variables ------------------------
	face_encoding, face_name = [], []
	
	#image load ------------------------------	
	for image in list(new_data):
		face_encoding.append(face_recognition.face_encodings(face_recognition.load_image_file(image+'.jpg'))[0])
		face_name.append(image.split('.')[0])
	
	# merging the lists ----------------------
	known_face_names += face_names
	known_face_encodings += face_encodings

	# change directory -----------------------
	chdir('../Data')
	# saving the data ------------------------
	DataFrame({
		'Name' : face_name, 
		'Data' : face_encoding
	}).to_csv(filename,header=False, mode='a', index=False)
	print('data update successful!')
# main directory ------------------------- 
chdir('..')

# break statement ------------------------
print('Press \'q\' to exit.....')

while True:
	# Grab a single frame of video -----------
	ret, frame = video_capture.read()

	# Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]

	# Only process every other frame of video to save time
	if process_this_frame:
		
		# Find all the faces and face encodings in the current frame of video
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names = []
		for face_encoding in face_encodings:

			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			name = "Unknown"

			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]
		   
				if name != face : 
					face = name
					time = datetime.today().strftime(" %I %M %p").replace(' 0', ' ')[:-1]+' M'
					
					gTTS(
						text=f"Welcome home {name}! it's, {time}.",
						lang= 'hi' #you can change the language from here
					).save(audio_file)

					system(f'mpg123 -q {audio_file}')

			face_names.append(name)
			
	process_this_frame = not process_this_frame	


	# Display the results
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		top *= 3
		right *= 4
		bottom *= 4
		left *= 4

		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Draw a label with a name below the face
		cv2.rectangle(frame, (left - 1, bottom + 35), (right - 30, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name.split()[0], (left + 6, bottom + 25), font, .7, (255, 255, 255), 1)

	# Display the resulting image
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Remove audio file
try:remove(audio_file)
except:pass

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
