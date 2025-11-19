from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import io
import zipfile

app = Flask(__name__)

@app.route("/split", methods=["POST"])
def split_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file trong request"}), 400

        file = request.files["file"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # --- PREPROCESS ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

        # Nối chữ + khung thành block lớn
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Bỏ contour nhỏ (đường chấm, đường kẻ)
            if w < 300 or h < 300:
                continue

            # CARE có chiều cao lớn
            if h < 600:  
                continue

            # CARE đứng → tỷ lệ cao/rộng lớn
            ratio = h / w
            if ratio < 1.2:
                continue

            blocks.append((x, y, w, h))

        # --- Chỉ giữ đúng 2 block → trái + phải ---
        blocks.sort(key=lambda b: b[0])  # sort theo X

        # Nếu dư hơn 2 thì chọn 2 block lớn nhất
        if len(blocks) > 2:
            blocks = sorted(blocks, key=lambda b: b[2] * b[3], reverse=True)[:2]
            blocks.sort(key=lambda b: b[0])

        # --- Xuất thành ZIP ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, (x, y, w, h) in enumerate(blocks, start=1):
                # Crop CARE + 250px dưới để lấy PCS
                y_end = min(y + h + 250, img.shape[0])
                crop = img[y:y_end, x:x + w]

                _, enc = cv2.imencode(".jpg", crop)
                zf.writestr(f"care_{i}.jpg", enc.tobytes())

        zip_buffer.seek(0)
        return send_file(zip_buffer, as_attachment=True, download_name="care_blocks.zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
