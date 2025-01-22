import cv2
import numpy as np
import os

def apply_transformations(image, mask, num_variations):
    images = []
    masks = []
    
    if image is None or mask is None:
        print("Error: Not possible to load Image !")
        return images, masks
    
    for _ in range(num_variations):
        # Random Rotation
        angle = np.random.uniform(-30, 30)  # Rotation zwischen -30 und 30 Grad
        h, w = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        
        # Rotation apply
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # Random shift
        tx = np.random.uniform(-20, 20)  # Horizontal verschieben
        ty = np.random.uniform(-20, 20)  # Vertikal verschieben
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply shift 
        translated_image = cv2.warpAffine(rotated_image, translation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        translated_mask = cv2.warpAffine(rotated_mask, translation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # Random  Scaling 
        scale = np.random.uniform(0.8, 1.2)
        scaled_image = cv2.resize(translated_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.resize(translated_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # Reshape  or Refill, back to 512x512 dimension
        final_image = cv2.resize(scaled_image, (512, 512), interpolation=cv2.INTER_LINEAR)
        final_mask = cv2.resize(scaled_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # add to list
        images.append(final_image)
        masks.append(final_mask)
    
    return images, masks

def save_variations(images, masks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (img, msk) in enumerate(zip(images, masks)):
        cv2.imwrite(os.path.join(output_dir, f"image_1variation_{i}.jpg"), img)
        cv2.imwrite(os.path.join(output_dir, f"image_1variation_{i}_MASK.jpg"), msk)

# Load Images
original_image = cv2.imread("/Users/dhil/Downloads/Gus-staining-python/pypotrace-master/tensorflow-test/Archive1/Skin/512##[0, 0, 238, 360, 255, 255]##1000110.jpg")  # Originalbild in Farbe (RGB)
original_mask  = cv2.imread("/Users/dhil/Downloads/Gus-staining-python/pypotrace-master/tensorflow-test/Archive1/Skin/512##[0, 0, 238, 360, 255, 255]##1000110_MASK.jpg", cv2.IMREAD_GRAYSCALE)  # Maske als Graustufenbild

# check if images are correct 
if original_image is None or original_mask is None:
    print("Error: Bild oder Maske konnten nicht geladen werden!")
else:
    # create 20 variants
    augmented_images, augmented_masks = apply_transformations(original_image, original_mask, num_variations=20)

    # Save Results
    save_variations(augmented_images, augmented_masks, output_dir="augmented_data")

