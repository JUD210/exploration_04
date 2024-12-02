# 필요한 라이브러리 임포트
from fastapi import FastAPI
import uvicorn
from fastapi.responses import StreamingResponse
from io import BytesIO
import torch
from PIL import Image
from pydantic import BaseModel

# Stable Diffusion 관련 라이브러리 임포트
from diffusers import StableDiffusionPipeline

# 디바이스 설정 (GPU가 사용 가능하면 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stable Diffusion 3.5 Medium 모델 로드
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
pipe.to(device)
pipe.enable_attention_slicing()  # 메모리 최적화

# FastAPI 앱 생성
app = FastAPI()

# 텍스트 입력 모델 정의
class TextInput(BaseModel):
    text: str

# 엔드포인트 정의
@app.post("/generate_image")
async def generate_image_endpoint(input: TextInput):
    text = input.text
    # 이미지 생성
    with torch.autocast(device.type):
        image = pipe(text).images[0]

    # 결과 이미지를 바이트 스트림으로 변환
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    # 결과 이미지를 응답으로 반환
    return StreamingResponse(buffered, media_type="image/png")

# 서버 실행
if __name__ == "__main__":
    uvicorn.run(
        "server_fastapi_rmbg:app",
        reload=True,  # 코드 변경 시 자동 리로드
        host="127.0.0.1",  # 로컬호스트
        port=12530,  # 포트
        log_level="info",  # 로깅 레벨
    )
