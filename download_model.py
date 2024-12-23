import requests

# Google Drive 다운로드 링크
file_url = "https://drive.google.com/uc?export=download&id=1kkl0H1xW3Lu3X-fCkLVU7UxlZl6-pluS"

# 파일 다운로드
response = requests.get(file_url)

# 다운로드한 파일 저장
with open("model_file.onnx", "wb") as f:
    f.write(response.content)

print("모델 파일 다운로드 완료!")
