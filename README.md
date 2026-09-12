# 注塑件表面缺陷 AI 检测系统 V5

本版本是在 V4 现有项目上直接升级，仍然使用 Gemini，新增两项核心能力：

1. **缺陷定位标记**：Gemini 在识别缺陷类型的同时返回缺陷边界框，网页自动在原图上绘制红框和编号。
2. **按缺陷类型自动分类整理**：检测成功后，原图和带标记图片会自动保存到对应缺陷目录，并在“分类整理”页中按类别查看。

## 新的检测流程

```text
上传图片
→ Gemini 判断主要缺陷类型
→ Gemini 返回所有明显缺陷的 box_2d
→ Pillow 在原图上绘制红框 + 编号
→ 显示每个框的类型 / 局部置信度 / 描述
→ 本地知识库给出候选原因与改进建议
→ 自动按主要缺陷类型归档
```

## 自动分类目录

第一次成功检测后，项目目录会自动出现：

```text
classified_images/
├─ 正常/
│  ├─ original/
│  └─ marked/
├─ 黑点/
├─ 白点/
├─ 划伤/
├─ 擦伤/
├─ 指印/
└─ index.csv
```

只有实际检测到的分类才会自动创建对应子文件夹。

- `original`：保存原图
- `marked`：保存带红框的标记图
- `index.csv`：分类索引

相同图片再次检测时，会按图片哈希更新原归档，避免大量重复文件。

## API Key 配置

继续沿用 V4 的方式，不在网页显示 API Key。

把 `.env.example` 复制为 `.env`，内容：

```text
GEMINI_API_KEY=你的真实APIKey
```

## 安装与启动

```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

如果已有 V4 的 `.venv`，可以直接在 V5 文件夹重新执行：

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 关于 Gemini 的缺陷框

Gemini 返回的 `box_2d` 使用：

```text
[ymin, xmin, ymax, xmax]
```

四个坐标归一化在 `0~1000`。程序会自动按上传图片的真实宽高换算成像素坐标，再用 Pillow 绘制红色边界框。

## 使用建议

Gemini 的边界框适合当前比赛原型和交互展示，但对非常细小、低对比度的工业缺陷，位置精度不一定稳定。后续如果你训练 YOLO，可以保留 V5 的网页、分类整理和知识库，只把 `analyze_image()` 替换成 YOLO 推理，其他页面逻辑基本可以继续复用。


## V8 真正原始字节版
- 原图预览改为浏览器 data URL 直接显示上传原始 bytes，绕过 st.image。
- 增加“100% 原始像素”查看。
- 显示文件分辨率、大小、SHA-256，并允许下载原始上传文件校验。
- Gemini 整图使用原始 bytes；18MB 以上使用 Files API。
- 2K/4K 可开启高清分块定位。
