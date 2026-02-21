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
            user_rank, used_score = engine.rank_from_input(
                stream=stream,
                score=score,
                rank=rank,
                art_mode="普高",
                art_culture_score=art_culture_score,
                art_special_score=art_special_score,
            )

            top_n = to_int(form["top_n"]) or 50
            top_n = max(10, min(top_n, 200))

            recommendations, tier_stats = engine.recommend(
                stream=stream,
                user_rank=user_rank,
                school_keyword=form["school_keyword"],
                major_keyword=form["major_keyword"],
                region_keyword=form["region_keyword"],
                top_n=top_n,
            )

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
