#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import json
import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, session
from openpyxl.utils import get_column_letter

from admission_service import AdmissionEngine, MOCK_SCORE_DIR, PROBABILITY_GUIDE, normalize_stream


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-in-production")
engine = AdmissionEngine(BASE_DIR)
ABOUT_CONTENT_PATH = os.path.join(BASE_DIR, "templates", "_about_content.html")
SITE_META_PATH = os.path.join(BASE_DIR, "site_meta.json")


def normalize_admin_path(path_value: str) -> str:
    value = str(path_value or "").strip()
    if not value:
        value = "/ops-console-kris-1027"
    if not value.startswith("/"):
        value = "/" + value
    value = re.sub(r"/+", "/", value)
    if value == "/":
        value = "/ops-console-kris-1027"
    return value


ADMIN_PATH = normalize_admin_path(os.environ.get("ADMIN_PATH", "/ops-console-kris-1027"))


def load_site_meta() -> dict:
    default = {"notice": "", "contact": ""}
    if not os.path.exists(SITE_META_PATH):
        return default
    try:
        with open(SITE_META_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        return {
            "notice": str(data.get("notice", "")).strip(),
            "contact": str(data.get("contact", "")).strip(),
        }
    except Exception:
        return default


def save_site_meta(meta: dict) -> None:
    payload = {
        "notice": str(meta.get("notice", "")).strip(),
        "contact": str(meta.get("contact", "")).strip(),
    }
    with open(SITE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@app.context_processor
def inject_site_meta():
    return {"site_meta": load_site_meta(), "admin_path": ADMIN_PATH}


def to_int(value: str) -> Optional[int]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return int(value)


def to_optional_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def score_display(value: Optional[float]):
    if value is None:
        return None
    if abs(value - round(value)) < 1e-6:
        return int(round(value))
    return round(value, 2)


def build_recommendation(stream: str, user_rank: int, form: dict):
    top_n = to_int(form.get("top_n", "")) or 50
    top_n = max(10, min(top_n, 200))
    recommendations, tier_stats = engine.recommend(
        stream=stream,
        user_rank=user_rank,
        school_keyword=form.get("school_keyword", ""),
        major_keyword=form.get("major_keyword", ""),
        region_keyword=form.get("region_keyword", ""),
        top_n=top_n,
    )
    return recommendations, tier_stats


def normalize_form_payload(defaults: dict, payload: dict) -> dict:
    form = defaults.copy()
    for key in form.keys():
        if key in payload and payload[key] is not None:
            form[key] = str(payload[key]).strip()
    return form


def get_admin_token() -> str:
    return os.environ.get("ADMIN_TOKEN", "").strip()


def is_admin_authed() -> bool:
    return bool(session.get("admin_ok"))


def safe_label(label: str) -> str:
    text = re.sub(r"\s+", " ", str(label or "").strip())
    text = re.sub(r"[^\w\u4e00-\u9fff\- ]", "", text)
    text = text.replace(" ", "_")
    return text[:40].strip("_")


def safe_text(value: object, max_length: int = 120) -> str:
    text = str(value if value is not None else "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_length:
        text = text[:max_length]
    return text


def build_export_rows(payload: dict) -> list:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("rows 必须是数组")
    if not raw_rows:
        raise ValueError("请至少选择 1 条志愿")
    if len(raw_rows) > 200:
        raise ValueError("单次最多导出 200 条志愿")

    stream = safe_text(payload.get("stream", ""), 20)
    user_rank = to_int(payload.get("user_rank", ""))
    source = safe_text(payload.get("source", ""), 20)

    rows = []
    for idx, item in enumerate(raw_rows, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx} 条志愿格式不正确")
        school_name = safe_text(item.get("school_name", ""), 120)
        major_name = safe_text(item.get("major_name", ""), 120)
        if not school_name or not major_name:
            raise ValueError(f"第 {idx} 条志愿缺少院校或专业名称")

        probability = to_optional_float(item.get("probability", ""))
        probability_display = ""
        if probability is not None:
            probability_display = f"{max(0.0, min(1.0, probability)) * 100:.1f}%"

        rows.append(
            {
                "序号": idx,
                "科类": stream,
                "院校代码": safe_text(item.get("school_code", ""), 40),
                "院校名称": school_name,
                "专业代码": safe_text(item.get("major_code", ""), 40),
                "专业名称": major_name,
                "最低分": safe_text(item.get("min_score", ""), 20),
                "最低位次": safe_text(item.get("min_rank", ""), 20),
                "位次比": safe_text(item.get("rank_ratio", ""), 20),
                "分档": safe_text(item.get("tier", ""), 8),
                "录取概率": probability_display,
                "输入位次": user_rank if user_rank is not None else "",
                "来源": source,
            }
        )
    return rows


@app.route("/", methods=["GET", "POST"])
def index():
    form = {
        "stream": "历史",
        "score": "",
        "rank": "",
        "top_n": "50",
        "art_culture_score": "",
        "art_special_score": "",
        "school_keyword": "",
        "major_keyword": "",
        "region_keyword": "",
    }
    result = None
    error = ""

    if request.method == "POST":
        for key in form.keys():
            form[key] = request.form.get(key, form[key]).strip()

        try:
            stream = normalize_stream(form["stream"])
            if not stream:
                raise ValueError("科类仅支持 历史/物理/艺术")

            score = to_optional_float(form["score"])
            rank = to_int(form["rank"])
            art_culture_score = to_optional_float(form["art_culture_score"])
            art_special_score = to_optional_float(form["art_special_score"])
            user_rank, used_score, _ = engine.rank_from_input(
                stream=stream,
                score=score,
                rank=rank,
                art_mode="普高",
                art_culture_score=art_culture_score,
                art_special_score=art_special_score,
            )

            recommendations, tier_stats = build_recommendation(stream, user_rank, form)
            result = {
                "stream": stream,
                "score": score,
                "used_score": used_score,
                "score_display": score_display(score),
                "used_score_display": score_display(used_score),
                "rank": rank,
                "user_rank": user_rank,
                "art_culture_score": art_culture_score,
                "art_special_score": art_special_score,
                "has_keyword_search": bool(
                    form["school_keyword"]
                    or form["major_keyword"]
                    or form["region_keyword"]
                ),
                "recommendations": recommendations,
                "tier_stats": tier_stats,
                "probability_guide": PROBABILITY_GUIDE,
            }
        except Exception as exc:
            error = str(exc)

    return render_template("index.html", form=form, result=result, error=error)


@app.route("/mock-convert", methods=["GET", "POST"])
def mock_convert():
    form = {
        "stream": "历史",
        "mock_exam_key": "",
        "mock_score": "",
        "mock_rank": "",
        "top_n": "50",
        "school_keyword": "",
        "major_keyword": "",
        "region_keyword": "",
    }
    mock_options = engine.get_mock_exam_options(form["stream"])
    result = None
    error = ""

    if request.method == "POST":
        for key in form.keys():
            form[key] = request.form.get(key, form[key]).strip()

        try:
            stream = normalize_stream(form["stream"])
            if stream not in {"历史", "物理"}:
                raise ValueError("联考换算仅支持 历史/物理")
            mock_options = engine.get_mock_exam_options(stream)

            mock_score = to_optional_float(form["mock_score"])
            mock_rank = to_int(form["mock_rank"])
            user_rank, used_score, conversion = engine.rank_from_input(
                stream=stream,
                score=None,
                rank=None,
                mock_score=mock_score,
                mock_rank=mock_rank,
                mock_exam_key=form["mock_exam_key"],
            )

            recommendations, tier_stats = build_recommendation(stream, user_rank, form)
            result = {
                "stream": stream,
                "used_score": used_score,
                "used_score_display": score_display(used_score),
                "mock_score_display": score_display(mock_score),
                "mock_rank": mock_rank,
                "user_rank": user_rank,
                "conversion": conversion,
                "has_keyword_search": bool(
                    form["school_keyword"]
                    or form["major_keyword"]
                    or form["region_keyword"]
                ),
                "recommendations": recommendations,
                "tier_stats": tier_stats,
                "probability_guide": PROBABILITY_GUIDE,
            }
        except Exception as exc:
            error = str(exc)

    return render_template("mock_convert.html", form=form, result=result, error=error, mock_options=mock_options)


@app.route("/api/mock-options", methods=["GET"])
def api_mock_options():
    stream = normalize_stream(request.args.get("stream", "历史"))
    if stream not in {"历史", "物理"}:
        return jsonify({"ok": False, "error": "科类仅支持 历史/物理"}), 400
    return jsonify({"ok": True, "data": engine.get_mock_exam_options(stream)})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    form = normalize_form_payload(
        {
            "stream": "历史",
            "score": "",
            "rank": "",
            "top_n": "50",
            "art_culture_score": "",
            "art_special_score": "",
            "school_keyword": "",
            "major_keyword": "",
            "region_keyword": "",
        },
        payload,
    )

    try:
        stream = normalize_stream(form["stream"])
        if not stream:
            raise ValueError("科类仅支持 历史/物理/艺术")

        score = to_optional_float(form["score"])
        rank = to_int(form["rank"])
        art_culture_score = to_optional_float(form["art_culture_score"])
        art_special_score = to_optional_float(form["art_special_score"])
        user_rank, used_score, _ = engine.rank_from_input(
            stream=stream,
            score=score,
            rank=rank,
            art_mode="普高",
            art_culture_score=art_culture_score,
            art_special_score=art_special_score,
        )

        recommendations, tier_stats = build_recommendation(stream, user_rank, form)
        data = {
            "stream": stream,
            "score": score,
            "used_score": used_score,
            "score_display": score_display(score),
            "used_score_display": score_display(used_score),
            "rank": rank,
            "user_rank": user_rank,
            "recommendations": recommendations,
            "tier_stats": tier_stats,
            "probability_guide": PROBABILITY_GUIDE,
        }
        return jsonify({"ok": True, "data": data})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/mock-convert", methods=["POST"])
def api_mock_convert():
    payload = request.get_json(silent=True) or {}
    form = normalize_form_payload(
        {
            "stream": "历史",
            "mock_exam_key": "",
            "mock_score": "",
            "mock_rank": "",
            "top_n": "50",
            "school_keyword": "",
            "major_keyword": "",
            "region_keyword": "",
        },
        payload,
    )

    try:
        stream = normalize_stream(form["stream"])
        if stream not in {"历史", "物理"}:
            raise ValueError("联考换算仅支持 历史/物理")

        mock_score = to_optional_float(form["mock_score"])
        mock_rank = to_int(form["mock_rank"])
        user_rank, used_score, conversion = engine.rank_from_input(
            stream=stream,
            score=None,
            rank=None,
            mock_score=mock_score,
            mock_rank=mock_rank,
            mock_exam_key=form["mock_exam_key"],
        )
        recommendations, tier_stats = build_recommendation(stream, user_rank, form)
        data = {
            "stream": stream,
            "used_score": used_score,
            "used_score_display": score_display(used_score),
            "mock_score_display": score_display(mock_score),
            "mock_rank": mock_rank,
            "user_rank": user_rank,
            "conversion": conversion,
            "recommendations": recommendations,
            "tier_stats": tier_stats,
            "probability_guide": PROBABILITY_GUIDE,
        }
        return jsonify({"ok": True, "data": data})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/export-wishlist", methods=["POST"])
def api_export_wishlist():
    payload = request.get_json(silent=True) or {}
    try:
        rows = build_export_rows(payload)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    df = pd.DataFrame(rows)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="志愿表")
        ws = writer.sheets["志愿表"]
        for idx, column in enumerate(df.columns, start=1):
            values = [str(column)] + [str(v) for v in df[column].tolist()]
            width = max(10, min(36, max(len(v) for v in values) + 2))
            ws.column_dimensions[get_column_letter(idx)].width = width

    output.seek(0)
    source = safe_label(payload.get("source", "预测")) or "预测"
    stream = safe_label(payload.get("stream", "志愿")) or "志愿"
    filename = f"{source}_{stream}_志愿表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route(ADMIN_PATH, methods=["GET", "POST"])
def admin():
    message = ""
    message_type = ""
    token = get_admin_token()
    admin_enabled = bool(token)
    authed = is_admin_authed()

    if request.method == "POST":
        action = request.form.get("action", "").strip()

        if action == "login":
            if not admin_enabled:
                message = "后台未启用：请先在 Railway 环境变量中设置 ADMIN_TOKEN。"
                message_type = "error"
            else:
                input_token = request.form.get("admin_token", "").strip()
                if input_token == token:
                    session["admin_ok"] = True
                    authed = True
                    message = "后台登录成功。"
                    message_type = "success"
                else:
                    message = "后台口令错误。"
                    message_type = "error"

        elif action == "logout":
            session.pop("admin_ok", None)
            authed = False
            message = "已退出后台。"
            message_type = "success"

        elif not authed:
            message = "请先登录后台。"
            message_type = "error"

        elif action == "save_site_meta":
            notice = request.form.get("notice", "")
            contact = request.form.get("contact", "")
            save_site_meta({"notice": notice, "contact": contact})
            message = "站点信息已保存。"
            message_type = "success"

        elif action == "save_about":
            content = request.form.get("about_content", "")
            with open(ABOUT_CONTENT_PATH, "w", encoding="utf-8") as f:
                f.write(content)
            message = "About 页面内容已更新。"
            message_type = "success"

        elif action == "upload_mock":
            stream = normalize_stream(request.form.get("stream", ""))
            if stream not in {"历史", "物理"}:
                message = "请选择历史或物理科类。"
                message_type = "error"
            else:
                upload = request.files.get("mock_file")
                if upload is None or upload.filename.strip() == "":
                    message = "请先选择要上传的联考文件。"
                    message_type = "error"
                elif not upload.filename.lower().endswith(".xlsx"):
                    message = "仅支持上传 .xlsx 文件。"
                    message_type = "error"
                else:
                    label = safe_label(request.form.get("mock_label", ""))
                    if not label:
                        origin = os.path.splitext(upload.filename)[0]
                        label = safe_label(origin) or "联考"
                    target_dir = os.path.join(BASE_DIR, MOCK_SCORE_DIR)
                    os.makedirs(target_dir, exist_ok=True)
                    filename = f"{stream}_{label}.xlsx"
                    save_path = os.path.join(target_dir, filename)
                    upload.save(save_path)
                    engine.reload()
                    message = f"联考文件已上传：{filename}"
                    message_type = "success"

        elif action == "delete_mock":
            stream = normalize_stream(request.form.get("stream", ""))
            key = request.form.get("mock_key", "").strip()
            options = {item["key"] for item in engine.get_mock_exam_options(stream)}
            if key not in options:
                message = "要删除的联考数据不存在。"
                message_type = "error"
            else:
                target_path = os.path.normpath(os.path.join(BASE_DIR, key))
                allowed_root = os.path.normpath(os.path.join(BASE_DIR, MOCK_SCORE_DIR))
                if target_path.startswith(allowed_root) and os.path.exists(target_path):
                    os.remove(target_path)
                    engine.reload()
                    message = "联考数据已删除。"
                    message_type = "success"
                else:
                    message = "删除失败：非法路径。"
                    message_type = "error"

    about_content = ""
    if os.path.exists(ABOUT_CONTENT_PATH):
        with open(ABOUT_CONTENT_PATH, "r", encoding="utf-8") as f:
            about_content = f.read()

    return render_template(
        "admin.html",
        admin_enabled=admin_enabled,
        authed=authed,
        message=message,
        message_type=message_type,
        site_meta=load_site_meta(),
        about_content=about_content,
        hist_mocks=engine.get_mock_exam_options("历史"),
        phys_mocks=engine.get_mock_exam_options("物理"),
    )


@app.route("/reload", methods=["POST"])
def reload_data():
    required_token = os.environ.get("RELOAD_TOKEN", "").strip()
    if required_token:
        token = request.headers.get("X-Admin-Token", "").strip()
        if token != required_token:
            return {"ok": False, "message": "unauthorized"}, 403
    try:
        engine.reload()
        return {"ok": True, "message": "数据已重新加载"}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}, 500


@app.route("/health", methods=["GET"])
def healthcheck():
    return {"ok": True}, 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
