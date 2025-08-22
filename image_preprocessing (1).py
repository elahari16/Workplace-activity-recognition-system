import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Part 1: Image Annotation
def annotate_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    clone = image.copy()
    annotations = []

    def draw_rectangle(event, x, y, flags, param):
        global ix, iy, drawing, annotations
        if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:  # End drawing
            drawing = False
            annotations.append((ix, iy, x, y))  # Save coordinates
            cv2.rectangle(clone, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", clone)

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", draw_rectangle)

    print("Press 's' to save and exit")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Save annotations
            break
    cv2.destroyAllWindows()

    # Save the image with annotations
    cv2.imwrite(output_path, clone)
    print("Annotations saved:", annotations)

# Example Usage
annotate_image("input.jpg", "annotated_image.jpg")

# Part 2: Data Augmentation
def augment_images(input_dir, output_dir, augment_count=5):
    # Create an instance of ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Loop through images in the input directory
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Convert the image to array format for augmentation
        image_array = np.expand_dims(image, axis=0)

        # Generate augmented images
        count = 0
        for batch in datagen.flow(image_array, batch_size=1, save_to_dir=output_dir, save_prefix="aug", save_format="jpeg"):
            count += 1
            if count >= augment_count:
                break  # Stop after generating augment_count images

# Example Usage
augment_images("input_images/", "augmented_images/", augment_count=10)
