import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from gradio_client import Client, file as gr_file
from PIL import Image
import requests
import io
from collections import deque

def analyze_image_background_complexity(image_path, padding=50):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' cannot be loaded. Please check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    top_left = gray[:padding, :padding]
    top_right = gray[:padding, w-padding:]
    bottom_left = gray[h-padding:, :padding]
    bottom_right = gray[h-padding:, w-padding:]

    padded_regions = np.concatenate((top_left.flatten(), top_right.flatten(), bottom_left.flatten(), bottom_right.flatten()))
    edges = cv2.Canny(padded_regions.reshape(padding * 4, -1), 100, 200)
    edge_count = np.sum(edges > 0)

    dft = cv2.dft(np.float32(padded_regions.reshape(padding * 4, -1)), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    high_freq_count = np.sum(magnitude_spectrum > np.percentile(magnitude_spectrum, 95))

    glcm = cv2.calcHist([padded_regions], [0], None, [256], [0, 256])
    texture_complexity = np.sum(glcm > np.percentile(glcm, 95))

    edge_threshold = 1000
    high_freq_threshold = 500
    texture_threshold = 2000

    is_complex = (
        edge_count > edge_threshold or
        high_freq_count > high_freq_threshold or
        texture_complexity > texture_threshold
    )

    result = 1 if is_complex else 0

    return result

client = Client("https://doevent-dis-background-removal.hf.space/--replicas/yzazc/")

def remove_background(image_path, size=(256, 256)): 
    result = client.predict(
                  gr_file(image_path),	# filepath  in 'image' Image component
                  api_name="/predict")
    
    # 반환된 결과에서 이미지 경로 추출
    result_image_path = result[1]['image']

    # 로컬 파일 경로에서 이미지를 엽니다.
    with open(result_image_path, 'rb') as f:
        image = Image.open(f)
        image.load()  # 파일을 메모리에 로드합니다.
    
    result_image = image.resize(size)

    return result_image

def generate_mask_from_image(image_path, threshold=50):
    def load_image(image_path, size=(256, 256)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size)
        return image

    def extract_background_color(image, padding=20, threshold=50):
        

        top = image[:padding, :, :]
        bottom = image[-padding:, :, :]
        left = image[:, :padding, :]
        right = image[:, -padding:, :]

        edges = np.concatenate((top.reshape(-1, 3), bottom.reshape(-1, 3),
                                left.reshape(-1, 3), right.reshape(-1, 3)), axis=0)

        black_pixels = np.sum(np.linalg.norm(edges, axis=1) < threshold)
        total_pixels = edges.shape[0]
        black_ratio = black_pixels / total_pixels

        if black_ratio >= 0.8:
            background_color = np.array([0, 0, 0])
        else:
            non_black_edges = edges[np.linalg.norm(edges, axis=1) >= threshold]
            background_color = np.median(non_black_edges, axis=0)

        return background_color, black_ratio

    def replace_black_corners(image, background_color, threshold=100):
        if image is None:
            raise ValueError("The image is not loaded properly.")

        rows, cols = image.shape[:2]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def is_valid(r, c):
            return 0 <= r < rows and 0 <= c < cols

        def is_near_black(pixel, threshold):
            return np.linalg.norm(pixel) < threshold

        def bfs(start_r, start_c):
            queue = deque([(start_r, start_c)])
            visited = set((start_r, start_c))
            to_change = []
            while queue:
                r, c = queue.popleft()
                if is_valid(r, c) and is_near_black(image[r, c], threshold):
                    to_change.append((r, c))
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if is_valid(nr, nc) and (nr, nc) not in visited:
                            if is_near_black(image[nr, nc], threshold):
                                queue.append((nr, nc))
                                visited.add((nr, nc))
            for r, c in to_change:
                image[r, c] = background_color

        corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
        for r, c in corners:
            if is_near_black(image[r, c], threshold):
                bfs(r, c)

        return image

    def preprocess_image(image, background_color, threshold=60):
        background_color = background_color.astype(np.uint8)
        background_image = np.full(image.shape, background_color, dtype=np.uint8)
        diff = cv2.absdiff(image, background_image)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
        return binary_image

    def create_mask(image, background_color, threshold=60):
        binary_image = preprocess_image(image, background_color, threshold)
        mask = np.zeros_like(binary_image)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        return mask

    image = load_image(image_path)

    h, w, _ = image.shape
    padding = int(min(h, w) * 0.15) # 15%의 패딩값 지정
    background_color, black_ratio = extract_background_color(image, padding)
    if black_ratio < 0.8:
        image = replace_black_corners(image, background_color)

    mask = create_mask(image, background_color, threshold)

    return mask


folder_path = '/home/ssu36/dowon/burgundy/datasets/mine/crab/train/good'
output_folder = '/home/ssu36/dowon/burgundy/datasets/mine/crab/DISthresh/good'

image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    try:
        complexity = analyze_image_background_complexity(image_path)
        print(f"{image_file}: Complexity = {complexity}")
    except ValueError as e:
        print(e)
        continue

    try:
        if complexity == 1:
            mask = remove_background(image_path)
        elif complexity == 0:
            # 게 데이터가 아닌 경우에는 복잡하지 않더라도 아래의 DIS 사용하기
            mask = generate_mask_from_image(image_path, 50)
            # mask = generate_mask_from_image(image_path, padding=20, threshold=50)
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        continue
    
    # 게 데이터가 아닌 경우에는 바로 아래대로 이름 설정
    mask_filename = image_file
    # mask_filename = image_file.replace(image_file.split('_')[0], 'thresh', 1)
    mask_path = os.path.join(output_folder, mask_filename)

    # 파일 확장자에 따라 저장 방식 결정
    file_extension = os.path.splitext(mask_path)[1].lower()
    if not file_extension:
        mask_path += '.JPG'

    if file_extension in ['.jpg', '.jpeg']:
        plt.imsave(mask_path, mask, cmap='gray', format='jpg')
    else:
        plt.imsave(mask_path, mask, cmap='gray')

print("Processing complete.")
