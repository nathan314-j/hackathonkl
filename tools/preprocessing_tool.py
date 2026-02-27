import cv2

def preprocess(image_path):
    image = cv2.imread(image_path)

    # 1. Resize (Increase resolution for better OCR detection)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert back to 3-channel BGR image so some OCR internals don't crash
    final_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return final_img