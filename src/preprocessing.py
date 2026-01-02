import cv2

def preprocess(images, size=(224, 224)):
    processed = []

    for img in images:
        resized = cv2.resize(img, size)
        processed.append(resized)

    print(f"Preprocessed {len(processed)} images.")
    return processed