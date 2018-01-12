from PIL import Image
import numpy as np

from MNISTTester import MNISTTester

# Load the model
mnist = MNISTTester(
            model_path='./models/mnist-cnn')

# Test it on a known image
test_img = Image.open('five.png').convert('L') # convert('L') converts to monochrome
number, accuracy = mnist.predict(test_img)
mnist.print_status('%d, accuracy: %f' % (number, accuracy))