import cv2
from train_model import new, view

def main():
    image_path = 'test_images/digit.jpg'
    padding_size = 10

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    new(image, padding_size)
    view(image, padding_size)

if __name__ == '__main__':
    main()