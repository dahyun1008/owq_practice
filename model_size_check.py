import os

def check_file_size(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return
    size = os.path.getsize(file_path) / (1024 * 1024)  # MB 단위
    print(f"File {file_path} size: {size:.2f} MB")

# .pth 파일 경로 지정
file_path = "opt-350m_3_01.pth"
check_file_size(file_path)
