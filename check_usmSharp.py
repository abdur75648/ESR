import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

from basicsr.utils import USMSharp
from PIL import Image, ImageFilter

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = T.ToTensor()
    return transform(image).unsqueeze(0)  # Add batch dimension

def save_image(tensor, output_path):
    transform = T.ToPILImage()
    image = transform(tensor.squeeze(0))  # Remove batch dimension
    image.save(output_path)

def unsharp_mask_opencv(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    # Step 1: Gaussian Blur
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    # Step 2: Create a mask by subtracting the blurred image from the original image
    mask = cv2.subtract(image, blurred)

    # Step 3: Add the mask to the original image
    sharpened = cv2.addWeighted(image, 1.0 + amount, mask, -amount, 0)

    # Step 4: Apply thresholding if necessary
    if threshold > 0:
        low_contrast_mask = np.absolute(mask) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened


def sharpen_laplacian(image):
    # Create a Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Convert back to uint8
    laplacian = cv2.convertScaleAbs(laplacian)

    # Add the Laplacian to the original image
    sharpened_image = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)

    return sharpened_image

def main():
    # input image
    input_image_path = "test_img_GT/1.jpg"

    # USM Sharp
    img = load_image(input_image_path)
    usm = USMSharp(radius=50, sigma=0)
    sharpened_img = usm(img, weight=0.5, threshold=10)
    save_image(sharpened_img, "1_sharpened_USM.jpg")

    # OpenCV
    img = cv2.imread(input_image_path)
    sharpened_image = unsharp_mask_opencv(img, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0)
    cv2.imwrite("1_sharpened_opencv.jpg", sharpened_image)

    # Laplacian
    img = cv2.imread(input_image_path)
    sharpened_image = sharpen_laplacian(img)
    cv2.imwrite("1_sharpened_laplacian.jpg", sharpened_image)

    # PIL Unsharp Mask
    image = Image.open(input_image_path)
    sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    sharpened_image.save("1_sharpened_pil.jpg")

if __name__ == "__main__":
    main()