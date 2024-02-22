import streamlit as st
from pathlib import Path
from PIL import Image
import os
import cv2
import sys

# streamlit 파일을 demo 폴더 안과 밖 모두 다 경로문제없이 실행하기 위해 path 추가
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir) 
sys.path.append(os.getcwd())
 
import pandas as pd
from PIL import Image, ImageOps

# 이미지 처리와 시각화
import numpy as np # 대규모 다차원 배열 및 행렬 연산을 위한 라이브러리
import matplotlib.pyplot as plt # 데이터 시각화 및 플로팅을 위한 라이브러리
from PIL import Image, ImageDraw, ImageFont # 이미지 처리를 위한 라이브러리, 그리고 이미지에 그리기, 텍스트 기능 추가 

# PaddleOCR 관련 모듈
from PaddleOCR.tools.infer.predict_det import TextDetector # 텍스트 탐지를 위한 클래스 
from PaddleOCR.tools.infer.predict_rec import TextRecognizer # 텍스트 탐지를 위한 클래스
import PaddleOCR.tools.infer.utility as utility # PaddleOCR 추론 유틸리티

def get_project_root() -> str:
    """Returns project root path.
 
    Returns
    -------
    str
        Project root path.
    """
    return str(Path(os.path.abspath(__file__)).parent)

st.set_page_config(layout="wide", page_title="7-segment display Scene-Text Recognition")

# 바운딩 박스의 색상 및 스타일 정보를 설정합니다.
box_color_RGBA  = (0,255,0,255) # 초록색 테두리
fill_color_RGBA = (0,255,0,50)  # 약간의 투명도를 가진 초록색 채우기

# 주어진 arguments를 파싱하여 TextDetector 객체를 생성합니다.
args = utility.parse_args()
args.det_model_dir = f'{os.path.dirname(get_project_root())}/PaddleOCR/output/det_inference' 
args.rec_char_dict_path = f'{os.path.dirname(get_project_root())}/PaddleOCR/ppocr/utils/en_dict.txt' 
args.rec_model_dir = f'{os.path.dirname(get_project_root())}/PaddleOCR/output/rec_inference'
text_detector = TextDetector(args) # Detection 모델을 불러옵니다 
text_recognizer = TextRecognizer(args) # Recognition 모델을 불러옵니다

### 이미지를 입력받아 Detection 부터 Recognition을 한번에 수행하는 모델을 만들어줍니다.  
class custom_ocr_model: 
    def __init__ (self, text_detector, text_recognizer): 
        """
        텍스트 검출 및 인식을 결합한 사용자 정의 OCR 모델 생성자.

        파라미터:
        - text_detector: 이미지에서 텍스트 영역을 검출하기 위한 모델
        - text_recognizer: 검출된 영역에서 텍스트를 인식하기 위한 모델
        """
        self.text_detector = text_detector 
        self.text_recognizer = text_recognizer 
    
    def predict_visualization(self, x):   
        """
        입력 이미지에 대해 텍스트 검출 및 인식을 수행합니다.

        파라미터:
        - x: 입력 이미지.

        반환값:
        - text: 이미지에서 인식된 텍스트.
        """   
        while True:
            width, height = x.size 
            if max(width, height) >500:
                x = x.resize((int(width/2), int(height/2)))
            else:
                break
        img = x.copy()
        draw = ImageDraw.Draw(img, 'RGBA')
        # 입력을 numpy 배열로 변환
        x = np.array(x) 
        
        # 텍스트 영역 검출
        dt_boxes, _ = text_detector(x) 
        if len(dt_boxes) != 0:
            # 각 검출된 영역의 높이(세로 길이)를 계산합니다.
            heights = dt_boxes[:, 2, 1] - dt_boxes[:, 0, 1]  
            # 가장 크게 검출된 박스의 인덱스를 찾아서 해당 박스만 선택합니다.
            max_height_index = np.argmax(heights)
            dt_boxes = dt_boxes[max_height_index] 

            # 선택된 영역(폴리곤)을 포함하는 최소 직사각형의 좌측 상단 및 우측 하단 좌표를 계산합니다.
            left_up_x = max(0,int(min(dt_boxes[:,0]) - 10)) # 숫자가 잘리는 부분이 발생하여, 10 만큼의 패딩을 추가함
            left_up_y = max(0,int(min(dt_boxes[:,1]) - 10)) 
            right_down_x = int(max(dt_boxes[:,0]) + 10)  
            right_down_y = int(max(dt_boxes[:,1]) + 10) 
            
            # 해당 직사각형 영역을 이미지에서 잘라냅니다. 이 영역에서 텍스트 인식이 수행될 것입니다.
            x = x[left_up_y:right_down_y,left_up_x:right_down_x] 
            draw.rectangle((left_up_x, left_up_y, right_down_x, right_down_y), outline=box_color_RGBA, fill=fill_color_RGBA, width = 3)
            text,proba = text_recognizer([x])[0][0]     # 텍스트 인식 
            text_position = (left_up_x -((right_down_y-left_up_y)/2), left_up_y - ((right_down_y-left_up_y)/2))  # 박스의 위쪽에 텍스트를 표시하기 위해 y 좌표를 조정
            draw.text(text_position, f'Recognized Text: {text}', fill=box_color_RGBA, font = ImageFont.truetype(f'{os.path.dirname(get_project_root())}/latin.ttf', int((right_down_y-left_up_y)/3))) # fill은 텍스트의 색상
            #img.show()
            #display(img)
            return img, text
        else:  
            # 검출된 텍스트 영역이 없으면, 그 사실을 사용자에게 알립니다.
            return img, "검출된 텍스트 영역이 없습니다."
            
cus_model = custom_ocr_model(text_detector, text_recognizer)



st.write("## 7-segment display Scene-Text Recognition")
st.write("이미지속 7-segment display를 찾아 숫자를 인식합니다.")
st.sidebar.write("## 이미지 업로드 :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

clear = False
def fix_image(upload):
    image = Image.open(upload)
    img, predicted_name = cus_model.predict_visualization(image)

    col1.write(f"이미지 추론 결과 : {predicted_name}")
    col1.image(img, use_column_width=False) 
    

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else: 
    clear = True
    # Example 이미지를 위한 컬럼
    example_images = [f'{get_project_root()}/examples/detection/image_data/01509.jpg', # 2020
                      f'{get_project_root()}/examples/detection/image_data/01511.jpg', # 3440
                      f'{get_project_root()}/examples/detection/image_data/01513.jpg', # 4650
                      f'{get_project_root()}/examples/detection/image_data/03850.jpg', # 7000
                      f'{get_project_root()}/examples/detection/image_data/03895.jpg'] # 6550
                    
    example_answer = ["2020", "3440", "4650", "7000", "6550"]
        

    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            # 예제 이미지를 위한 버튼 추가
            #st.write(f"샘플 이미지 {idx}")
            col.image(example_images[idx])
            if st.button(f"샘플 이미지 {idx+1} 테스트(클릭)", key=f"btn_{idx}"):
                # 선택된 이미지를 fix_image 함수로 전달
                example_image_path = example_images[idx]
                image = Image.open(example_image_path)
                img, predicted_name = cus_model.predict_visualization(image)
                col1.write(f"이미지 추론 결과 : {predicted_name}")
                col1.image(img, use_column_width=False) 
                        