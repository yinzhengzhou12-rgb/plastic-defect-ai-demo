import csv
import hashlib
import io
import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field


st.set_page_config(
    page_title="注塑件表面缺陷 AI 检测系统",
    page_icon="🔍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "history.csv"
CLASSIFIED_DIR = BASE_DIR / "classified_images"
CLASSIFICATION_INDEX = CLASSIFIED_DIR / "index.csv"
CLASS_NAMES = ["正常", "黑点", "白点", "划伤", "擦伤", "指印"]

# 从 .env 或 Windows 环境变量中读取 GEMINI_API_KEY。
# 网页中不会显示密钥输入框。
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

with open(BASE_DIR / "knowledge.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)


class LocalizedDefect(BaseModel):
    defect_type: Literal["黑点", "白点", "划伤", "擦伤", "指印"]
    confidence: int = Field(ge=0, le=100)
    box_2d: list[int] = Field(
        description="缺陷边界框 [ymin, xmin, ymax, xmax]，坐标归一化到 0-1000"
    )
    description: str = Field(description="该框中可见缺陷的简短描述")


class DefectResult(BaseModel):
    defect_type: Literal["正常", "黑点", "白点", "划伤", "擦伤", "指印"]
    severity: Literal["轻微", "一般", "严重"]
    confidence: int = Field(ge=0, le=100)
    basis: str
    defects: list[LocalizedDefect] = Field(
        description="图片中全部可见缺陷的位置列表；若正常则返回空列表"
    )


def ensure_history_file():
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "检测时间", "文件名", "产品型号", "材料",
                "缺陷类型", "严重程度", "置信度", "视觉判断依据",
                "候选原因", "改进建议"
            ])


def save_history(filename, product_model, material, result, diagnosis):
    ensure_history_file()
    causes_text = "；".join([x["cause"] for x in diagnosis["causes"]])
    actions_text = "；".join(diagnosis["actions"])
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            product_model,
            material,
            result.defect_type,
            result.severity,
            result.confidence,
            result.basis,
            causes_text,
            actions_text,
        ])


def read_history():
    ensure_history_file()
    with open(HISTORY_FILE, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def ensure_classification_index():
    CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    if not CLASSIFICATION_INDEX.exists():
        with open(CLASSIFICATION_INDEX, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "检测时间", "图片哈希", "原文件名", "产品型号", "材料",
                "主要缺陷类型", "全部缺陷类型", "缺陷数量", "严重程度",
                "置信度", "原图路径", "标记图路径"
            ])


def read_classification_index():
    ensure_classification_index()
    with open(CLASSIFICATION_INDEX, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_classification_index(rows):
    ensure_classification_index()
    fieldnames = [
        "检测时间", "图片哈希", "原文件名", "产品型号", "材料",
        "主要缺陷类型", "全部缺陷类型", "缺陷数量", "严重程度",
        "置信度", "原图路径", "标记图路径"
    ]
    with open(CLASSIFICATION_INDEX, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", stem)
    return stem[:80] or "image"


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def normalized_box_to_pixels(box_2d, width, height):
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return None

    try:
        ymin, xmin, ymax, xmax = [float(x) for x in box_2d]
    except (TypeError, ValueError):
        return None

    ymin = max(0.0, min(1000.0, ymin))
    xmin = max(0.0, min(1000.0, xmin))
    ymax = max(0.0, min(1000.0, ymax))
    xmax = max(0.0, min(1000.0, xmax))

    if ymax < ymin:
        ymin, ymax = ymax, ymin
    if xmax < xmin:
        xmin, xmax = xmax, xmin

    left = int(round(xmin / 1000.0 * width))
    top = int(round(ymin / 1000.0 * height))
    right = int(round(xmax / 1000.0 * width))
    bottom = int(round(ymax / 1000.0 * height))

    if right - left < 2 or bottom - top < 2:
        return None

    return left, top, right, bottom


def annotate_image(image: Image.Image, result: DefectResult) -> Image.Image:
    """在原图上画缺陷框。框左上角仅标编号，避免中文字体兼容问题。"""
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    width, height = marked.size
    line_width = max(3, int(round(min(width, height) * 0.004)))
    tag_h = max(24, line_width * 5)
    tag_w = max(28, line_width * 6)

    for idx, defect in enumerate(result.defects, 1):
        box = normalized_box_to_pixels(defect.box_2d, width, height)
        if box is None:
            continue

        left, top, right, bottom = box
        draw.rectangle([left, top, right, bottom], outline=(255, 30, 30), width=line_width)

        # 红色编号标签。具体编号和缺陷类型在网页表格中对应显示。
        tag_left = left
        tag_top = max(0, top - tag_h)
        tag_right = min(width, tag_left + tag_w)
        tag_bottom = min(height, tag_top + tag_h)
        draw.rectangle([tag_left, tag_top, tag_right, tag_bottom], fill=(255, 30, 30))
        draw.text((tag_left + 6, tag_top + 4), str(idx), fill=(255, 255, 255))

    return marked


def localized_defect_rows(result: DefectResult):
    rows = []
    for idx, defect in enumerate(result.defects, 1):
        rows.append({
            "编号": idx,
            "缺陷类型": defect.defect_type,
            "置信度": f"{defect.confidence}%",
            "位置 [ymin,xmin,ymax,xmax]": str(defect.box_2d),
            "局部特征": defect.description,
        })
    return rows


def archive_detection(image_bytes, image, marked_image, filename, product_model, material, result):
    """按主要缺陷类型自动归档原图与标记图；相同图片哈希再次检测时更新原归档。"""
    ensure_classification_index()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    rows = read_classification_index()

    # 删除同一图片旧归档，避免反复检测生成大量重复文件。
    kept = []
    for row in rows:
        if row.get("图片哈希") == image_hash:
            for key in ("原图路径", "标记图路径"):
                rel = row.get(key, "")
                if rel:
                    try:
                        (BASE_DIR / rel).unlink(missing_ok=True)
                    except OSError:
                        pass
        else:
            kept.append(row)

    category = result.defect_type
    category_dir = CLASSIFIED_DIR / category
    original_dir = category_dir / "original"
    marked_dir = category_dir / "marked"
    original_dir.mkdir(parents=True, exist_ok=True)
    marked_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{safe_stem(filename)}_{image_hash[:10]}"
    original_path = original_dir / f"{base_name}.png"
    marked_path = marked_dir / f"{base_name}_marked.png"

    image.convert("RGB").save(original_path, format="PNG")
    marked_image.convert("RGB").save(marked_path, format="PNG")

    all_types = sorted({d.defect_type for d in result.defects})
    if not all_types and result.defect_type != "正常":
        all_types = [result.defect_type]
    if result.defect_type == "正常":
        all_types = ["正常"]

    row = {
        "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "图片哈希": image_hash,
        "原文件名": filename,
        "产品型号": product_model,
        "材料": material,
        "主要缺陷类型": result.defect_type,
        "全部缺陷类型": "、".join(all_types),
        "缺陷数量": str(len(result.defects)),
        "严重程度": result.severity,
        "置信度": str(result.confidence),
        "原图路径": str(original_path.relative_to(BASE_DIR)),
        "标记图路径": str(marked_path.relative_to(BASE_DIR)),
    }

    kept.append(row)
    write_classification_index(kept)
    return row


def analyze_image(image: Image.Image) -> tuple[DefectResult, str]:
    if not API_KEY:
        raise RuntimeError(
            "未检测到 GEMINI_API_KEY。请在项目目录的 .env 文件或 Windows 环境变量中配置。"
        )

    client = genai.Client(api_key=API_KEY)

    prompt = """
你是一名注塑件外观质量检测助手。请同时完成“缺陷分类”和“缺陷定位”。
只根据上传图片中可见的表面特征做判断，不要虚构不可见的工艺信息。

允许的缺陷类型只有：
正常、黑点、白点、划伤、擦伤、指印。

请严格遵守：
1. defect_type：整张图片的主要缺陷类型，只能从上述六类选择一个；若未发现明显缺陷则为“正常”。
2. severity：只能为“轻微 / 一般 / 严重”。
3. confidence：0-100 整数，表示对整张图主要结论的置信度。
4. basis：用简短中文说明整张图的视觉判断依据。
5. defects：列出图片中所有明显缺陷位置。如果判断为正常，必须返回空列表 []。
6. defects 中每个元素：
   - defect_type：只能是 黑点、白点、划伤、擦伤、指印，不允许写“正常”；
   - confidence：0-100 整数；
   - box_2d：必须是 [ymin, xmin, ymax, xmax]，四个值都归一化到 0-1000；
   - description：简短描述这个框内看到了什么。
7. 边界框应尽量紧贴缺陷本身，不要框住整件产品，也不要把大面积正常区域包含进去。
8. 同一张图如存在多个缺陷，请分别给出多个框；不要只定位一个。
9. 图片不清晰、缺陷不明显或位置不确定时，应降低置信度。
"""

    # 保持现有项目的模型回退逻辑。
    model_candidates = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
    ]

    retryable = ("503", "UNAVAILABLE", "high demand", "429", "RESOURCE_EXHAUSTED")
    skippable = ("404", "NOT_FOUND", "no longer available")
    errors = []

    for model_name in model_candidates:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DefectResult,
                    ),
                )

                if getattr(response, "parsed", None) is not None:
                    parsed = response.parsed
                    if isinstance(parsed, DefectResult):
                        return parsed, model_name
                    return DefectResult.model_validate(parsed), model_name

                return DefectResult.model_validate_json(response.text), model_name

            except Exception as e:
                msg = str(e)
                errors.append(f"{model_name} 第{attempt+1}次：{msg}")

                if any(x in msg for x in skippable):
                    break

                if any(x in msg for x in retryable):
                    if attempt < 2:
                        time.sleep(1.5 * (2 ** attempt))
                        continue
                    break

                raise

    raise RuntimeError(
        "当前 Gemini 服务暂时无法完成请求。最近错误：\n" + "\n".join(errors[-4:])
    )


def make_seed(image_bytes: bytes, defect_type: str, severity: str, material: str, model: str, nonce: int):
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(defect_type.encode("utf-8"))
    h.update(severity.encode("utf-8"))
    h.update(material.encode("utf-8"))
    h.update(model.encode("utf-8"))
    h.update(str(nonce).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big")


def build_diagnosis(defect_type, severity, image_bytes, material="", product_model="", nonce=0):
    """
    原因和建议全部来自本地知识库，不再额外调用 API。
    同一缺陷的不同图片会得到不同组合；
    点击“换一组建议”也可在不消耗 API 的情况下生成另一组。
    """
    item = knowledge[defect_type]
    rng = random.Random(
        make_seed(image_bytes, defect_type, severity, material, product_model, nonce)
    )

    causes = list(item["causes"])

    if defect_type == "正常":
        picked = causes[:1]
    else:
        rng.shuffle(causes)
        picked = []
        used_categories = set()

        for cause in causes:
            if cause["category"] not in used_categories:
                picked.append(cause)
                used_categories.add(cause["category"])
            if len(picked) == 3:
                break

        if len(picked) < 3:
            for cause in causes:
                if cause not in picked:
                    picked.append(cause)
                if len(picked) == 3:
                    break

    actions = []

    for cause in picked:
        options = list(cause["actions"])
        rng.shuffle(options)
        if options:
            actions.append(options[0])

    general_actions = list(item.get("general_actions", []))
    rng.shuffle(general_actions)
    for action in general_actions:
        if action not in actions:
            actions.append(action)
            break

    return {
        "causes": picked,
        "actions": actions[:4],
    }


def make_text_report(filename, product_model, material, result, diagnosis, used_model):
    cause_lines = []
    for i, x in enumerate(diagnosis["causes"], 1):
        cause_lines.append(
            f"{i}. [{x['category']}] {x['cause']}\n"
            f"   建议验证：{x['check']}"
        )

    action_lines = [f"{i}. {x}" for i, x in enumerate(diagnosis["actions"], 1)]

    if result.defects:
        location_lines = []
        for i, d in enumerate(result.defects, 1):
            location_lines.append(
                f"{i}. {d.defect_type}，局部置信度 {d.confidence}%："
                f"box_2d={d.box_2d}；{d.description}"
            )
        location_text = "\n".join(location_lines)
    else:
        location_text = "未检测到需要标记的明显缺陷区域。"

    return f"""注塑件表面缺陷 AI 检测报告

检测时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
图片文件：{filename}
产品型号：{product_model or "未填写"}
材料：{material or "未填写"}

一、AI 视觉检测
缺陷类型：{result.defect_type}
严重程度：{result.severity}
置信度：{result.confidence}%
视觉判断依据：{result.basis}

二、缺陷位置
{location_text}

三、候选原因（需现场验证）
{chr(10).join(cause_lines)}

四、改进与验证建议
{chr(10).join(action_lines)}

说明：
1. 缺陷类别与位置由视觉 AI 辅助判断，边界框坐标为 0-1000 归一化坐标。
2. 候选原因和改进建议来自本地注塑缺陷知识库。
3. 仅凭照片不能确认真实工艺根因，应结合材料、机台、模具和工艺参数复核。
4. 本次 AI 模型：{used_model}
"""


st.title("注塑件表面缺陷 AI 检测系统")
st.caption("AI 自动识别 + 缺陷定位标记 + 动态原因分析 + 自动分类整理 + 历史记录 + 报告导出")

tab1, tab2, tab3, tab4 = st.tabs(["单张检测", "批量检测", "分类整理", "历史记录"])


with tab1:
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.subheader("上传图片")
        uploaded_file = st.file_uploader(
            "JPG / JPEG / PNG",
            type=["jpg", "jpeg", "png"],
            key="single_upload",
        )
        image = None
        image_bytes = b""
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image).convert("RGB")
            st.image(image, caption="原始图片", use_container_width=True)

    with c2:
        st.subheader("产品信息")
        material = st.text_input("材料（可选）", placeholder="例如：ABS", key="single_material")
        product_model = st.text_input("产品型号（可选）", placeholder="例如：A-001", key="single_model")

        detect = st.button(
            "开始 AI 检测",
            type="primary",
            use_container_width=True,
            disabled=image is None,
            key="single_detect",
        )

    if detect:
        try:
            with st.spinner("AI 正在识别缺陷类型并定位缺陷位置……"):
                result, used_model = analyze_image(image)
                marked_image = annotate_image(image, result)
                marked_bytes = image_to_png_bytes(marked_image)
                archive_row = archive_detection(
                    image_bytes,
                    image,
                    marked_image,
                    uploaded_file.name,
                    product_model,
                    material,
                    result,
                )

            st.session_state["last_detection"] = {
                "filename": uploaded_file.name,
                "image_hash": hashlib.sha256(image_bytes).hexdigest(),
                "image_bytes": image_bytes,
                "marked_bytes": marked_bytes,
                "material": material,
                "product_model": product_model,
                "result": result.model_dump(),
                "used_model": used_model,
                "nonce": 0,
                "archive_row": archive_row,
            }
        except Exception as e:
            st.error("AI 调用失败。")
            st.code(str(e))

    last = st.session_state.get("last_detection")

    if last and uploaded_file is not None:
        current_hash = hashlib.sha256(image_bytes).hexdigest()

        if last["image_hash"] == current_hash:
            result = DefectResult.model_validate(last["result"])
            marked_image = Image.open(io.BytesIO(last["marked_bytes"])).convert("RGB")

            st.divider()
            st.header("检测结果")

            st.subheader("缺陷定位标记")
            st.image(
                marked_image,
                caption="红框及编号为 Gemini 返回的缺陷位置；编号与下表对应",
                use_container_width=True,
            )

            if result.defects:
                st.dataframe(localized_defect_rows(result), use_container_width=True, hide_index=True)
            else:
                st.success("未检测到需要标记的明显缺陷区域。")

            st.download_button(
                "下载带缺陷标记的图片（PNG）",
                data=last["marked_bytes"],
                file_name=f"{safe_stem(last['filename'])}_marked.png",
                mime="image/png",
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("主要缺陷类型", result.defect_type)
            m2.metric("严重程度", result.severity)
            m3.metric("AI 置信度", f"{result.confidence}%")
            m4.metric("定位缺陷数", len(result.defects))

            st.caption(
                f"已自动分类归档到：classified_images/{result.defect_type}/  "
                "（原图保存在 original，标记图保存在 marked）"
            )

            st.subheader("视觉判断依据")
            st.write(result.basis)

            if st.button(
                "换一组候选原因 / 改进建议（不再次调用 AI）",
                key="reroll_single",
            ):
                st.session_state["last_detection"]["nonce"] += 1
                last = st.session_state["last_detection"]

            diagnosis = build_diagnosis(
                result.defect_type,
                result.severity,
                last["image_bytes"],
                last["material"],
                last["product_model"],
                last["nonce"],
            )

            left, right = st.columns(2)

            with left:
                st.subheader("候选原因（需验证）")
                st.caption("不是仅凭照片确认的根因；系统从本地知识库中动态组合。")
                for i, x in enumerate(diagnosis["causes"], 1):
                    with st.container(border=True):
                        st.markdown(f"**{i}. {x['cause']}**")
                        st.caption(f"类别：{x['category']}")
                        st.write(f"验证方法：{x['check']}")

            with right:
                st.subheader("改进与验证建议")
                for i, x in enumerate(diagnosis["actions"], 1):
                    st.markdown(f"**{i}.** {x}")

            report_text = make_text_report(
                last["filename"],
                last["product_model"],
                last["material"],
                result,
                diagnosis,
                last["used_model"],
            )

            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "下载本次检测报告（TXT）",
                    data=report_text.encode("utf-8-sig"),
                    file_name=f"检测报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with b2:
                if st.button("保存本次结果到历史记录", use_container_width=True):
                    save_history(
                        last["filename"],
                        last["product_model"],
                        last["material"],
                        result,
                        diagnosis,
                    )
                    st.success("已保存到历史记录。")


with tab2:
    st.subheader("批量检测")
    st.write("一次选择多张图片。检测后会自动按主要缺陷类型归档到 classified_images。")

    batch_files = st.file_uploader(
        "选择多张图片",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    batch_material = st.text_input("统一材料（可选）", key="batch_material")
    batch_model = st.text_input("统一产品型号（可选）", key="batch_model")

    batch_detect = st.button(
        "开始批量检测",
        type="primary",
        disabled=not batch_files,
        key="batch_detect",
    )

    if batch_detect:
        rows = []
        progress = st.progress(0)

        for idx, file in enumerate(batch_files, 1):
            try:
                file_bytes = file.getvalue()
                img = Image.open(io.BytesIO(file_bytes))
                img = ImageOps.exif_transpose(img).convert("RGB")
                result, used_model = analyze_image(img)
                marked_img = annotate_image(img, result)
                diagnosis = build_diagnosis(
                    result.defect_type,
                    result.severity,
                    file_bytes,
                    batch_material,
                    batch_model,
                    0,
                )

                archive_detection(
                    file_bytes,
                    img,
                    marked_img,
                    file.name,
                    batch_model,
                    batch_material,
                    result,
                )

                all_types = sorted({x.defect_type for x in result.defects})
                if not all_types:
                    all_types = [result.defect_type]

                rows.append({
                    "文件名": file.name,
                    "主要缺陷类型": result.defect_type,
                    "全部检测类型": "、".join(all_types),
                    "缺陷数量": len(result.defects),
                    "严重程度": result.severity,
                    "置信度": f"{result.confidence}%",
                    "候选原因": "；".join([x["cause"] for x in diagnosis["causes"]]),
                    "改进建议": "；".join(diagnosis["actions"]),
                })

                save_history(
                    file.name,
                    batch_model,
                    batch_material,
                    result,
                    diagnosis,
                )
            except Exception as e:
                rows.append({
                    "文件名": file.name,
                    "主要缺陷类型": "调用失败",
                    "全部检测类型": "",
                    "缺陷数量": "",
                    "严重程度": "",
                    "置信度": "",
                    "候选原因": "",
                    "改进建议": str(e),
                })

            progress.progress(idx / len(batch_files))

        st.dataframe(rows, use_container_width=True, hide_index=True)

        successful = [x for x in rows if x["主要缺陷类型"] in CLASS_NAMES]
        if successful:
            counts = Counter(x["主要缺陷类型"] for x in successful)
            st.subheader("本批次分类统计")
            st.dataframe(
                [{"缺陷类型": k, "图片数量": counts.get(k, 0)} for k in CLASS_NAMES if counts.get(k, 0)],
                use_container_width=True,
                hide_index=True,
            )

        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

            st.download_button(
                "下载批量检测结果（CSV）",
                data=output.getvalue().encode("utf-8-sig"),
                file_name=f"批量检测结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


with tab3:
    st.subheader("分类整理")
    classified_rows = read_classification_index()

    if not classified_rows:
        st.info("目前还没有分类归档图片。完成一次单张或批量检测后会自动生成。")
    else:
        counts = Counter(row["主要缺陷类型"] for row in classified_rows)
        summary_rows = [
            {"缺陷类型": name, "图片数量": counts.get(name, 0)}
            for name in CLASS_NAMES
        ]
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

        available = [name for name in CLASS_NAMES if counts.get(name, 0) > 0]
        selected_category = st.selectbox("查看某一分类", available)
        selected_rows = [
            row for row in classified_rows
            if row["主要缺陷类型"] == selected_category
        ]
        selected_rows.sort(key=lambda x: x.get("检测时间", ""), reverse=True)

        max_show = min(30, len(selected_rows))

        # 当该分类只有 1 张图片时，Streamlit 的 slider 不允许
        # min_value 和 max_value 相等，因此直接显示 1 张，不创建滑块。
        if max_show <= 1:
            show_count = max_show
        else:
            show_count = st.slider(
                "显示最近多少张",
                min_value=1,
                max_value=max_show,
                value=min(12, max_show),
            )

        st.caption(
            f"目录：classified_images/{selected_category}/original 和 "
            f"classified_images/{selected_category}/marked"
        )

        gallery_rows = selected_rows[:show_count]
        for start in range(0, len(gallery_rows), 3):
            cols = st.columns(3)
            for col, row in zip(cols, gallery_rows[start:start + 3]):
                with col:
                    marked_path = BASE_DIR / row["标记图路径"]
                    if marked_path.exists():
                        st.image(
                            str(marked_path),
                            caption=(
                                f"{row['原文件名']}\n"
                                f"{row['主要缺陷类型']} | {row['严重程度']} | {row['置信度']}%"
                            ),
                            use_container_width=True,
                        )
                    st.caption(
                        f"检测时间：{row['检测时间']}  |  "
                        f"全部类型：{row['全部缺陷类型']}"
                    )

        category_csv = io.StringIO()
        if selected_rows:
            writer = csv.DictWriter(category_csv, fieldnames=selected_rows[0].keys())
            writer.writeheader()
            writer.writerows(selected_rows)
            st.download_button(
                f"下载 {selected_category} 分类清单（CSV）",
                data=category_csv.getvalue().encode("utf-8-sig"),
                file_name=f"{selected_category}_分类清单.csv",
                mime="text/csv",
            )


with tab4:
    st.subheader("历史记录")
    rows = read_history()

    if not rows:
        st.info("目前还没有保存的检测历史。")
    else:
        st.dataframe(rows[::-1], use_container_width=True)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

        st.download_button(
            "下载全部历史记录（CSV）",
            data=output.getvalue().encode("utf-8-sig"),
            file_name="history.csv",
            mime="text/csv",
        )
