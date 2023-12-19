import cv2
import sys
import uuid
import os

if len(sys.argv) != 3:
    print(str(len(sys.argv)) + ' args provided')
    sys.exit('Usage: python ' + sys.argv[0] + ' SrcImgDir DstFaceDir')

imagePath = sys.argv[1]
facePath = sys.argv[2]
base_dir = os.path.dirname(__file__)

if not os.path.exists(base_dir + facePath):
	os.makedirs(base_dir + facePath)

count = 0
for file in os.listdir(base_dir + imagePath):
	file_name, file_extension = os.path.splitext(file)
	if (file_extension in ['.png','.jpg']):
		image = cv2.imread(base_dir + imagePath + '/' + file)
		print("Processing: " + base_dir + imagePath + '/' + str(file))
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
		faces = faceCascade.detectMultiScale(
		    gray,
		    scaleFactor=1.3,
		    minNeighbors=3,
 		    minSize=(50, 50)
		)
		print("Found {0} Faces".format(len(faces)))

		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			roi_color = image[y:y + h, x:x + w]
			faceFile = str(uuid.uuid4()) + '.jpg'
			count += 1
			cv2.imwrite(base_dir + facePath + '/' + faceFile, roi_color)
			print("Saving 1 face: ", facePath + '/' + faceFile)

print("Extracted " + str(count) + " faces from all images")