import cv2

# Load image
img = cv2.imread("car.jpg")
if img is None:raise Exception("Image not loaded")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load cascade
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Detect plates
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Draw rectangles
for (x, y, w, h) in plates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show image
cv2.imshow('plates', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
