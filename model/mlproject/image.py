import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Load the pre-trained parameters
params = np.load("model.npz")
W1, b1, W2, b2, W3, b3 = params["W1"], params["b1"], params["W2"], params["b2"], params["W3"], params["b3"]

def relu(Z):
    # ReLU activation function
    return np.maximum(Z, 0)

def predict_classes(output_activations):
    # Predict the class labels
    return np.argmax(output_activations, axis=0)

def softmax(Z):
    # Softmax activation function
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=0)

def forward_propagation(W1, b1, W2, b2, W3, b3, X):
    # Perform forward propagation through the neural network
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def process_image(image_path, target_size=(28, 28)):
    # Load and preprocess the image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title("Original Image")
    plt.show()
    # Resize the image
    img_resized = img.resize(target_size)
    
    # Convert the image to grayscale
    img_gray = img_resized.convert('L')
    
    # Equalize the histogram of the grayscale image
    img_gray = ImageOps.equalize(img_gray)
    
    # Convert the image to a numpy array
    img_array = np.array(img_gray)
    
    # Reshape the array to 1D array of size 784
    img_1d = img_array.reshape(-1)
    
    # Thresholding and inversion
    img_1d = np.where(img_1d > 20, 255, img_1d)
    img_1d = np.where(1, 255 - img_1d, 0)
    
    # Convert to column vector and normalize
    column_vector = img_1d.reshape(-1, 1)
    normalized_vector = column_vector.astype(float) / 255.0
    
    # Display the processed image
    img_array=np.where(img_array>20,255,img_array)
    plt.imshow(img_array, cmap='gray')
    plt.title("Processed Image")
    plt.show()
    
    # Forward propagation through the neural network
    _, _, _, _, _, output_activations = forward_propagation(W1, b1, W2, b2, W3, b3, normalized_vector)
    
    # Predict and return the class
    return predict_classes(output_activations)[0]

# Process the image and print the predicted class
print(process_image("Seven.jpg"))
