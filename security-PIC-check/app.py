from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# 設定模板資料夾
TEMPLATE_FOLDER = "templates"

def compare_images(template_path, target_image):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # 確保尺寸相同
    template = cv2.resize(template, (target_gray.shape[1], target_gray.shape[0]))

    # 計算 SSIM 相似度
    score, _ = ssim(template, target_gray, full=True)

    return score

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "沒有檔案"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "未選擇檔案"}), 400

    # 讀取上傳的圖像
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 對比所有模板
    for template_file in os.listdir(TEMPLATE_FOLDER):
        template_path = os.path.join(TEMPLATE_FOLDER, template_file)
        score = compare_images(template_path, img)

        if score > 0.8:  # 設定 SSIM 相似度門檻值
            return jsonify({"message": f"找到真品！匹配度: {score:.2f}"})

    return jsonify({"message": "此圖疑似非真品"}), 200

if __name__ == '__main__':
    app.run(debug=True)
