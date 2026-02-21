#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import Optional

from flask import Flask, render_template, request

from admission_service import AdmissionEngine, PROBABILITY_GUIDE, normalize_stream


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
engine = AdmissionEngine(BASE_DIR)


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


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


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
