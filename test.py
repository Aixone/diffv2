import fitz
import re
import os
import json
from PIL import Image
import io
import base64

# 打开 PDF 文件
pdf_path = "./1.pdf"
doc = fitz.open(pdf_path)

# 输出路径
image_dir = "extracted_images"
os.makedirs(image_dir, exist_ok=True)

multimodal_data = []
formula_id = 0
image_id = 0

# 定义公式识别特征（含变量、运算符、约束符号）
math_pattern = re.compile(r'([xX]\d*|\d+\.?\d*|≤|≥|=|\+|\-|\*|\\|∈|→|∀|∃|\|\||min|max|\^|√|∞)')

for page_number in range(len(doc)):
    page = doc[page_number]

    # 提取文本并按段落分开
    blocks = page.get_text("blocks")  # 返回 (x0, y0, x1, y1, "text", block_no)
    for block in blocks:
        text = block[4].strip()
        if not text:
            continue
        lines = text.split('\n')
        for line in lines:
            if math_pattern.search(line) and len(line) < 150:
                formula_id += 1
                multimodal_data.append({
                    "type": "text",
                    "id": f"formula_{formula_id:03}",
                    "page": page_number + 1,
                    "content": line
                })

    # 提取图片
    images = page.get_images(full=True)
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        img_filename = f"page{page_number+1}_img{img_index+1}.png"
        image.save(os.path.join(image_dir, img_filename))

        # base64 编码保存
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_id += 1
        multimodal_data.append({
            "type": "image",
            "id": f"img_{image_id:03}",
            "page": page_number + 1,
            "file_name": img_filename,
            "base64_data": image_b64
        })

# 保存为 JSON 模拟数据库
with open("multimodal_extracted.json", "w", encoding="utf-8") as f:
    json.dump(multimodal_data, f, ensure_ascii=False, indent=2)

print(f"✅ 完成提取，共提取公式 {formula_id} 条，图片 {image_id} 张。")