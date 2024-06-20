import os
import numpy as np
import cv2
from collections import Counter

# 파일 이름 변경(_가 있을 경우)
def filename(target_folder):
    # 파일 리스트 가져오기
    files = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

    # 파일 이름 변경
    i = 0
    for file_name in sorted(image_files):
        # 기존 파일 이름에서 _를 포함한 _ 앞에 있는 문자를 제거
        base_name, ext = os.path.splitext(file_name)
        if '_' in base_name:
            new_base_name = base_name.split('_', 1)[1]  # _ 뒤의 부분만 사용
        else:
            new_base_name = base_name  # _가 없으면 그대로 사용
        new_name = f"{new_base_name}{ext}"
        src = os.path.join(target_folder, file_name)
        dst = os.path.join(target_folder, new_name)
        os.rename(src, dst)
        i += 1

    print(i)
    print("파일 이름 변경이 완료되었습니다.")


# 파일 이름 변경(순서 no 상관)
def rename_images_in_folder(folder_path):
    # 이미지 확장자 목록
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg'}
    
    # 폴더 내의 파일 목록 가져오기
    files = [file for file in os.listdir(folder_path) if os.path.splitext(file)[1].lower() in image_extensions]
    
    # 파일 정렬
    files.sort()
    
    # 파일 이름 변경
    for index, file in enumerate(files):
        extension = os.path.splitext(file)[1].lower()
        new_name = f"{index:05d}{extension}"
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
        print(f"Renamed '{file}' to '{new_name}'")


# 파일 확장자 개수 보기
def print_image_extensions_count(folder_path):
    # 이미지 확장자 목록
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg'}
    
    # 폴더 내 파일 확장자 목록과 개수 계산
    extensions_count = Counter(
        os.path.splitext(file)[1].lower() 
        for file in os.listdir(folder_path) 
        if os.path.splitext(file)[1].lower() in image_extensions
    )
    
    # 콘솔에 개수 출력
    for extension, count in extensions_count.items():
        print(f"{extension}: {count}")


# 더미 마스크 만들기
def generate_and_save_random_masks(folder_path, n, size=(256, 256), num_pixels=30):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(n):
        # Generate a single region mask
        mask = np.zeros(size, dtype=np.uint8)
        max_attempts = 1000  # Maximum attempts to find a contiguous block
        attempts = 0

        while attempts < max_attempts:
            x = np.random.randint(0, size[0])
            y = np.random.randint(0, size[1])

            if x + num_pixels // 2 < size[0] and y + num_pixels // 2 < size[1]:
                mask[x:x + num_pixels // 2, y:y + num_pixels // 2] = 1
                break
            
            attempts += 1

        # Save the mask with a 5-digit filename
        file_name = str(i).zfill(5) + '.png'
        file_path = os.path.join(folder_path, file_name)
        cv2.imwrite(file_path, mask * 255)  # Saving the mask image as PNG



def delete_analyze_image_background_complexity(source_folder, target_folder = None, target_folder_2 = None, padding=50):
    # 소스 폴더 내의 모든 파일 순회
    for filename in os.listdir(source_folder):
        source_image_path = os.path.join(source_folder, filename)

        # 이미지 파일만 처리
        if not (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')):
            continue

        # 이미지 불러오기
        image = cv2.imread(source_image_path)
        if image is None:
            print(f"Image at path '{source_image_path}' cannot be loaded. Skipping...")
            continue

        # 이미지를 회색조로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 네 구역에서 패딩된 부분 추출
        top_left = gray[:padding, :padding]
        top_right = gray[:padding, w-padding:]
        bottom_left = gray[h-padding:, :padding]
        bottom_right = gray[h-padding:, w-padding:]

        # 네 구역을 하나로 합침
        padded_regions = np.concatenate((top_left.flatten(), top_right.flatten(), bottom_left.flatten(), bottom_right.flatten()))

        # 엣지 검출
        edges = cv2.Canny(padded_regions.reshape(padding * 4, -1), 100, 200)
        edge_count = np.sum(edges > 0)

        # 고속 푸리에 변환
        dft = cv2.dft(np.float32(padded_regions.reshape(padding * 4, -1)), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        high_freq_count = np.sum(magnitude_spectrum > np.percentile(magnitude_spectrum, 95))

        # 텍스처 복잡도 계산
        glcm = cv2.calcHist([padded_regions], [0], None, [256], [0, 256])
        texture_complexity = np.sum(glcm > np.percentile(glcm, 95))

        # 임계값 설정
        edge_threshold = 1000
        high_freq_threshold = 500
        texture_threshold = 2000

        # 복잡도 여부 판단
        is_complex = (
            edge_count > edge_threshold or
            high_freq_count > high_freq_threshold or
            texture_complexity > texture_threshold
        )

        result = 1 if is_complex else 0

        # 이미지 삭제
        if result == 1:
            os.remove(source_image_path)
            print(f"Image '{source_image_path}' has been deleted due to complex background.")

            # 타겟 폴더에서 동일한 이름의 이미지 삭제
            if target_folder != None:
                target_image_path = os.path.join(target_folder, filename)
                if os.path.exists(target_image_path):
                    os.remove(target_image_path)
            if target_folder_2 != None:
                target_folder_2 = os.path.join(target_folder, filename)
                if os.path.exists(target_folder_2):
                    os.remove(target_folder_2)


# 대상 폴더 경로 설정
source_folder = '/home/ssu36/dowon/burgundy/datasets/mine/crab/test/good'
target_folder = ''
target_folder_2 = ''
delete_analyze_image_background_complexity(source_folder)