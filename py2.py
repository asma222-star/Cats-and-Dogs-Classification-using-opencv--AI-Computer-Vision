import onnxruntime as ort
import cv2
import numpy as np
import os

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None, None

    resized = cv2.resize(img, (224, 224))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_float = rgb_img.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))  # (C, H, W)
    return img, np.expand_dims(img_chw, axis=0)

def classify_image(session, input_name, output_name, input_data, labels, batch_size):
    if batch_size > 1:
        input_data = np.tile(input_data, (batch_size, 1, 1, 1))

    outputs = session.run([output_name], {input_name: input_data})[0]
    prediction = outputs[0]
    class_id = int(np.argmax(prediction))
    confidence = float(prediction[class_id])
    label = f"{labels[class_id]}: {confidence * 100:.2f}%"
    return label

def main():
    model_path = r'C:\Users\Garnet\Desktop\python folder\py\cats_v_dogs.onnx'
    images_folder = r'C:\Users\Garnet\Desktop\python folder\py\cane'
    output_folder = r'classified_outputs'
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_shape = session.get_inputs()[0].shape
    batch_size = expected_shape[0] if isinstance(expected_shape[0], int) else 1
    labels = ['Dog', 'Cat']

    for file in os.listdir(images_folder):
        image_path = os.path.join(images_folder, file)
        if not (file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.png')):
            continue

        original_img, input_data = preprocess_image(image_path)
        if input_data is None:
            continue

        label = classify_image(session, input_name, output_name, input_data, labels, batch_size)
        print(f"{file}: {label}")

        # Draw label on image
        cv2.putText(original_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        output_path = os.path.join(output_folder, f"classified_{file}")
        cv2.imwrite(output_path, original_img)

    print(f"All images classified. Results saved to: {output_folder}")

if __name__ == "__main__":
    main()
