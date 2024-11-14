import cv2
import numpy as np
import os
from PIL import Image
import pillow_heif
from pathlib import Path

def convert_heic_to_numpy(heic_path):
    """Convert HEIC image to numpy array."""
    heif_file = pillow_heif.read_heif(heic_path)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
    )
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def find_card_contour(image):
    """Find the contour of the card in the image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get image area
        image_area = image.shape[0] * image.shape[1]
        
        # Filter contours by area
        valid_contours = [cnt for cnt in contours 
                         if 0.05 * image_area < cv2.contourArea(cnt) < 0.95 * image_area]
        
        if valid_contours:
            return max(valid_contours, key=cv2.contourArea)
    
    return None

def get_card_corners(contour):
    """Get the corners of the card from its contour."""
    # Approximate the contour to a polygon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    # If we get 4 points, use them directly
    if len(approx) == 4:
        return approx.reshape(4, 2)
    
    # Otherwise, find the minimum area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)

def order_points(pts):
    """Order points in clockwise order starting from top-left."""
    # Convert points to float32
    pts = pts.astype(np.float32)
    
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def crop_and_transform(image, corners):
    """Crop and transform the card to a standard size."""
    # Order the corners
    rect = order_points(corners)
    
    # Calculate width and height based on the detected corners
    widthA = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    widthB = np.sqrt(((rect[3][0] - rect[2][0]) ** 2) + ((rect[3][1] - rect[2][1]) ** 2))
    heightA = np.sqrt(((rect[2][0] - rect[1][0]) ** 2) + ((rect[2][1] - rect[1][1]) ** 2))
    heightB = np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2))
    
    # Take maximum width and height
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    # Ensure aspect ratio is maintained
    if maxWidth > maxHeight:
        maxHeight = int(maxWidth * 1.4)  # Standard playing card ratio
    else:
        maxWidth = int(maxHeight / 1.4)
    
    # Define destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Perform the perspective transform
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def remove_background(image):
    """Remove the black background while preserving card colors."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the black background
    # Adjust these values if needed
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Invert mask to get the card area
    mask = cv2.bitwise_not(mask)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Create white background
    white_background = np.ones_like(image) * 255
    
    # Blend the original image with white background
    result = image.copy()
    result[mask == 0] = white_background[mask == 0]
    
    return result

def process_image(input_path, output_path):
    """Process a single image."""
    try:
        # Read and convert HEIC image
        image = convert_heic_to_numpy(input_path)
        
        # Find card contour
        contour = find_card_contour(image)
        if contour is None:
            print(f"Failed to find card in {input_path}")
            return False
        
        # Get corners
        corners = get_card_corners(contour)
        
        # Crop and transform
        transformed = crop_and_transform(image, corners)
        
        # Remove black background
        result = remove_background(transformed)
        
        # Enhance contrast slightly
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Save the result
        cv2.imwrite(output_path, result)
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def main():
    input_base = Path("C:/Users/jobri/OneDrive - Drexel University/MEM679/cards_data")
    output_base = Path("C:/Users/jobri/OneDrive - Drexel University/MEM679/cropped_cards_data")
    
    output_base.mkdir(exist_ok=True)
    
    for card_folder in input_base.iterdir():
        if card_folder.is_dir():
            output_folder = output_base / card_folder.name
            output_folder.mkdir(exist_ok=True)
            
            for image_file in card_folder.glob("*.heic"):
                output_path = output_folder / (image_file.stem + ".jpg")
                
                if process_image(str(image_file), str(output_path)):
                    print(f"Successfully processed: {image_file}")
                else:
                    print(f"Failed to process: {image_file}")

if __name__ == "__main__":
    main()