import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('mnist_model.h5')  

# Function to preprocess the image
def preprocess_image(img):
    # Resize the image to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert the colors (because MNIST digits are white on black background)
    img = cv2.bitwise_not(img)
    # Reshape the image to match the input shape of the model
    img = img.reshape(1, 28, 28, 1)
    # Normalize the pixel values
    img = img.astype('float32') / 255.0
    return img

# Function to predict the digit using the model
def predict_digit(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Predict the digit using the model
    prediction = model.predict(processed_image)
    # Get the predicted digit
    predicted_digit = np.argmax(prediction)
    return predicted_digit

# Capture an image using the camera
camera = cv2.VideoCapture(0)  # Change the parameter if you have multiple cameras

while True:
    ret, frame = camera.read()
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)
    if key == 32:  # Press Spacebar to capture the image
        # Predict the digit
        digit = predict_digit(frame)
        print("Predicted Digit:", digit)
    elif key == 27:  # Press Escape to exit
        break

camera.release()
cv2.destroyAllWindows()

# Run the cameratest.py script and point the camera at a handwritten digit. Press the Spacebar to capture the image and see the predicted digit printed in the terminal. Press the Escape key to exit the program.
# take the mirror image from the camera capture

cv2.flip(frame, 1)



