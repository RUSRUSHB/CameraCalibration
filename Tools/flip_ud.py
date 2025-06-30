import cv2
import os

# Get all image files in current directory
img_files = [f for f in os.listdir('./IntrinsicCalibration/data/gray_same/') if f.endswith(('.jpg', '.png', '.jpeg'))]

print(img_files)
for img_file in img_files:
    # Read image
    img = cv2.imread('./IntrinsicCalibration/data/gray_same/' + img_file)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Flip image vertically
    flipped = cv2.flip(img, 0)  # 0 means flip around x-axis (vertical flip)
    
    # Save flipped image with 'flip_' prefix
    new_filename = img_file
    cv2.imwrite(new_filename, flipped)
