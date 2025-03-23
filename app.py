from flask import Flask, request, jsonify
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import os
template_path = os.path.join(os.getcwd(), "template_folder", "template.jpg")
template = cv2.imread(template_path)
print("Reading template from:", template_path)  # 確認實際路徑


app = Flask(__name__)

def compare_images(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 重新調整尺寸以確保一致
    gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    score, _ = ssim(gray1, gray2, full=True)
    return score

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"message": "未收到圖片"}), 400
    
    file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # 載入 template 圖片
    template = cv2.imread("template_folder/template.jpg")  # 請確保雲端有這個檔案
    
    similarity = compare_images(template, image)
    threshold = 0.85  # 設定相似度門檻
    
    if similarity >= threshold:
        return jsonify({"message": "找到真品，Voronoi 圖案匹配成功", "similarity": similarity})
    else:
        return jsonify({"message": "此圖疑似非真品", "similarity": similarity})

if __name__ == "__main__":
    app.run(debug=True)
