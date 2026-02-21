#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


SAFE_RATIO = 0.90
STEADY_RATIO = 1.10

PROBABILITY_GUIDE = [
    {"min": 0.75, "label": "高把握", "desc": "基本属于保底区间，可优先考虑专业偏好"},
    {"min": 0.60, "label": "较稳妥", "desc": "录取机会较高，适合作为稳妥志愿"},
    {"min": 0.45, "label": "可尝试", "desc": "存在机会，建议与保底志愿搭配"},
    {"min": 0.30, "label": "冲刺项", "desc": "风险偏高，建议放在前段冲刺"},
    {"min": 0.00, "label": "高风险", "desc": "录取不确定性较大，谨慎填报"},
]

STREAM_FILES = {
    "历史": {
        "admission": ["hist.xlsx"],
        "score_table": "历史类含加分一分段表.xlsx",
    },
    "物理": {
        "admission": ["phys.xlsx"],
        "score_table": "物理类含加分一分段表.xlsx",
    },
    "艺术": {
        "admission": ["艺术类.xlsx", "艺术批.xlsx"],
        "score_table": None,
    },
}


def normalize_col(col: str) -> str:
    return re.sub(r"\s+", "", str(col))


def normalize_stream(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip().lower()
    if v in {"历史", "hist", "history", "1", "h"}:
        return "历史"
    if v in {"物理", "phys", "physics", "2", "p"}:
        return "物理"
    if v in {"艺术", "art", "arts", "3", "a"}:
        return "艺术"
    return ""


def resolve_existing_file(base_dir: str, candidates: List[str]) -> str:
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"未找到数据文件: {', '.join(candidates)}")


def load_admission(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到录取表文件: {path}")

    df = pd.read_excel(path)
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})

    rename_map = {
        "投档最低分": "最低分",
        "最低分": "最低分",
        "院校代号": "院校代号",
        "院校名称": "院校名称",
        "专业代号": "专业代号",
        "专业名称": "专业名称",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required = ["院校代号", "院校名称", "专业代号", "专业名称", "最低分"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"录取表缺少列: {', '.join(miss)}")

    df = df[required].dropna(how="all")
    df["最低分"] = pd.to_numeric(df["最低分"], errors="coerce")
    df = df.dropna(subset=["最低分"])

    for col in ["院校代号", "院校名称", "专业代号", "专业名称"]:
        df[col] = df[col].astype(str).str.strip().replace({"nan": ""})

    df = df[(df["院校名称"] != "") & (df["专业名称"] != "")]
    return df.reset_index(drop=True)


def load_score_table(path: str) -> Tuple[Dict[float, int], Optional[float], Optional[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到一分一段表文件: {path}")

    df = pd.read_excel(path)
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})

    if "分数段" not in df.columns or "累计人数" not in df.columns:
        raise ValueError("一分一段表需要包含列: 分数段, 累计人数")

    score_to_rank_map: Dict[float, int] = {}
    top_score = None
    top_rank = None

    for _, row in df.iterrows():
        segment_raw = row["分数段"]
        if pd.isna(segment_raw):
            continue
        segment = str(segment_raw).strip()
        cum = pd.to_numeric(row["累计人数"], errors="coerce")
        if pd.isna(cum):
            continue
        rank = int(cum)
        if segment == "" or segment.lower() == "nan":
            continue

        if isinstance(segment_raw, (int, float)) and not isinstance(segment_raw, bool):
            score_to_rank_map[float(segment_raw)] = rank
            continue

        cleaned_segment = segment.replace("分", "").replace(" ", "")

        if "及以上" in cleaned_segment:
            nums = re.findall(r"\d+(?:\.\d+)?", cleaned_segment)
            if nums:
                top_score = float(nums[0])
                top_rank = rank
            continue

        if re.fullmatch(r"\d+(?:\.\d+)?", cleaned_segment):
            score_to_rank_map[float(cleaned_segment)] = rank
            continue

        nums = re.findall(r"\d+(?:\.\d+)?", cleaned_segment)
        if nums:
            score_to_rank_map[float(nums[-1])] = rank

    if not score_to_rank_map:
        raise ValueError("一分一段表没有有效分数段数据")
    return score_to_rank_map, top_score, top_rank


def score_to_rank(score: float, score_map: Dict[float, int], top_score: Optional[float], top_rank: Optional[int]) -> int:
    if top_score is not None and top_rank is not None and score >= top_score:
        return top_rank

    sorted_scores = sorted(score_map.keys(), reverse=True)
    for s in sorted_scores:
        if score >= s:
            return int(score_map[s])
    return int(score_map[sorted_scores[-1]])


def calc_ratio(user_rank: int, min_rank: int) -> float:
    if min_rank <= 0:
        return 999.0
    return float(user_rank) / float(min_rank)


def calc_tier(ratio: float) -> str:
    if ratio <= SAFE_RATIO:
        return "保"
    if ratio <= STEADY_RATIO:
        return "稳"
    return "冲"


def calc_probability(ratio: float) -> float:
    if ratio <= 0.60:
        p = 0.82
    elif ratio <= SAFE_RATIO:
        p = 0.60 + (SAFE_RATIO - ratio) * (0.22 / 0.30)
    elif ratio <= STEADY_RATIO:
        p = 0.35 + (STEADY_RATIO - ratio) * (0.25 / 0.20)
    elif ratio <= 1.40:
        p = 0.15 + (1.40 - ratio) * (0.20 / 0.30)
    else:
        p = 0.08
    p = max(0.05, min(0.90, p))
    return round(float(p), 4)


def round_half_up(value: float, digits: int = 2) -> float:
    if digits <= 0:
        quant = "1"
    else:
        quant = "0." + "0" * (digits - 1) + "1"
    return float(Decimal(str(value)).quantize(Decimal(quant), rounding=ROUND_HALF_UP))


def calc_art_composite(art_mode: str, culture_score: float, art_score: float) -> float:
    _ = art_mode
    value = (culture_score / 750.0) * 300.0 * 0.5 + art_score * 0.5
    return round_half_up(float(value), 2)


def build_rank_map_from_admission_scores(admission_df: pd.DataFrame) -> Tuple[Dict[float, int], Optional[float], Optional[int]]:
    score_counts = admission_df["最低分"].value_counts().sort_index(ascending=False)
    score_map: Dict[float, int] = {}
    cumulative = 0
    for score, count in score_counts.items():
        cumulative += int(count)
        score_map[float(score)] = cumulative
    if not score_map:
        raise ValueError("艺术类录取表没有有效最低分数据")
    top_score = max(score_map.keys())
    top_rank = score_map[top_score]
    return score_map, top_score, top_rank


def serialize_score(value: Union[float, int]) -> Union[float, int]:
    number = float(value)
    if abs(number - round(number)) < 1e-6:
        return int(round(number))
    return round(number, 2)


def explain_probability(probability: float, tier: str, ratio: float) -> str:
    for level in PROBABILITY_GUIDE:
        if probability >= level["min"]:
            return f"{level['label']}：{level['desc']}（分档{tier}，位次比{ratio:.3f}）"
    return f"高风险：录取不确定性较大（分档{tier}，位次比{ratio:.3f}）"


def build_ranked_admission(admission_df: pd.DataFrame, score_map: Dict[float, int], top_score: Optional[float], top_rank: Optional[int]) -> pd.DataFrame:
    df = admission_df.copy()
    df["最低位次"] = df["最低分"].apply(lambda x: score_to_rank(float(x), score_map, top_score, top_rank))
    return df


class AdmissionEngine:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.datasets: Dict[str, Dict[str, object]] = {}
        self.reload()

    def reload(self) -> None:
        for stream, paths in STREAM_FILES.items():
            admission_path = resolve_existing_file(self.data_dir, paths["admission"])
            admission_df = load_admission(admission_path)
            if paths["score_table"]:
                score_path = os.path.join(self.data_dir, paths["score_table"])
                score_map, top_score, top_rank = load_score_table(score_path)
            else:
                score_map, top_score, top_rank = build_rank_map_from_admission_scores(admission_df)
            ranked_admission = build_ranked_admission(admission_df, score_map, top_score, top_rank)
            self.datasets[stream] = {
                "admission": ranked_admission,
                "score_map": score_map,
                "top_score": top_score,
                "top_rank": top_rank,
            }

    def rank_from_input(
        self,
        stream: str,
        score: Optional[float],
        rank: Optional[int],
        art_mode: str = "普高",
        art_culture_score: Optional[float] = None,
        art_special_score: Optional[float] = None,
    ) -> Tuple[int, Optional[float]]:
        stream = normalize_stream(stream)
        if stream not in self.datasets:
            raise ValueError("科类仅支持 历史/物理/艺术")
        used_score: Optional[float] = None
        if rank is not None and rank > 0:
            return int(rank), used_score

        if stream == "艺术":
            if score is not None:
                used_score = float(score)
            else:
                if art_culture_score is None or art_special_score is None:
                    raise ValueError("艺术类请填写综合分，或填写文化成绩与艺考成绩")
                used_score = calc_art_composite(art_mode, float(art_culture_score), float(art_special_score))
        else:
            if score is None:
                raise ValueError("请至少输入分数或位次")
            used_score = float(score)

        data = self.datasets[stream]
        return score_to_rank(used_score, data["score_map"], data["top_score"], data["top_rank"]), used_score

    def recommend(
        self,
        stream: str,
        user_rank: int,
        school_keyword: str = "",
        major_keyword: str = "",
        region_keyword: str = "",
        top_n: int = 30,
    ) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
        stream = normalize_stream(stream)
        if stream not in self.datasets:
            raise ValueError("科类仅支持 历史/物理/艺术")

        df = self.datasets[stream]["admission"].copy()
        df["位次比"] = df["最低位次"].apply(lambda x: calc_ratio(user_rank, int(x)))
        df["分档"] = df["位次比"].apply(calc_tier)
        df["概率"] = df["位次比"].apply(calc_probability)
        df["贴合度"] = (df["位次比"] - 1.0).abs()

        if school_keyword:
            df = df[df["院校名称"].str.contains(school_keyword, na=False)]
        if major_keyword:
            df = df[df["专业名称"].str.contains(major_keyword, na=False)]
        if region_keyword:
            df = df[df["院校名称"].str.contains(region_keyword, na=False)]

        top_n = max(1, int(top_n))
        has_keyword = bool(school_keyword or major_keyword or region_keyword)

        df = df.sort_values(by=["贴合度", "概率", "最低位次"], ascending=[True, False, True])

        if has_keyword:
            df = df.head(top_n)
        else:
            rows = []
            school_count: Dict[str, int] = {}
            per_school_limit = 2
            for _, row in df.iterrows():
                school = str(row["院校名称"])
                count = school_count.get(school, 0)
                if count >= per_school_limit:
                    continue
                rows.append(row)
                school_count[school] = count + 1
                if len(rows) >= top_n:
                    break
            if rows:
                df = pd.DataFrame(rows)
            else:
                df = df.head(top_n)

        stats = {"冲": 0, "稳": 0, "保": 0}
        for tier, count in df["分档"].value_counts().to_dict().items():
            stats[tier] = int(count)

        rows: List[Dict[str, object]] = []
        for _, row in df.iterrows():
            ratio = round(float(row["位次比"]), 4)
            tier = str(row["分档"])
            probability = round(float(row["概率"]), 4)
            rows.append(
                {
                    "school_code": row["院校代号"],
                    "school_name": row["院校名称"],
                    "major_code": row["专业代号"],
                    "major_name": row["专业名称"],
                    "min_score": serialize_score(row["最低分"]),
                    "min_rank": int(row["最低位次"]),
                    "rank_ratio": ratio,
                    "tier": tier,
                    "probability": probability,
                    "probability_desc": explain_probability(probability, tier, ratio),
                }
            )
        return rows, stats

    def parse_wishlist_text(self, text: str) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if re.search(r"[,\uff0c]", line):
                parts = re.split(r"[,\uff0c]", line, maxsplit=1)
            else:
                parts = re.split(r"\s+", line, maxsplit=1)
            parts = [p.strip() for p in parts]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                items.append({"type": "invalid", "line": line, "school": "", "major": ""})
                continue
            school = parts[0]
            major = parts[1]
            if re.fullmatch(r"[0-9A-Za-z]+", school) and re.fullmatch(r"[0-9A-Za-z]+", major):
                item_type = "code"
            else:
                item_type = "name"
            items.append({"type": item_type, "line": line, "school": school, "major": major})
        return items

    def evaluate_wishlist(self, stream: str, user_rank: int, items: List[Dict[str, str]]) -> Dict[str, object]:
        stream = normalize_stream(stream)
        if stream not in self.datasets:
            raise ValueError("科类仅支持 历史/物理/艺术")

        admission_df = self.datasets[stream]["admission"].copy()
        admission_df["位次比"] = admission_df["最低位次"].apply(lambda x: calc_ratio(user_rank, int(x)))
        admission_df["分档"] = admission_df["位次比"].apply(calc_tier)
        admission_df["概率"] = admission_df["位次比"].apply(calc_probability)

        rows: List[Dict[str, object]] = []
        for idx, item in enumerate(items, start=1):
            if item["type"] == "invalid":
                rows.append(
                    {
                        "index": idx,
                        "input": item["line"],
                        "matched": False,
                        "message": "格式错误，需包含院校和专业两列",
                    }
                )
                continue

            if item["type"] == "code":
                matched = admission_df[
                    (admission_df["院校代号"] == item["school"]) & (admission_df["专业代号"] == item["major"])
                ]
            else:
                matched = admission_df[
                    admission_df["院校名称"].str.contains(item["school"], na=False)
                    & admission_df["专业名称"].str.contains(item["major"], na=False)
                ]

            if matched.empty:
                rows.append(
                    {
                        "index": idx,
                        "input": item["line"],
                        "matched": False,
                        "message": "未匹配到院校专业，请检查名称或代码",
                    }
                )
                continue

            best = matched.sort_values(by=["概率", "位次比"], ascending=[False, True]).iloc[0]
            rows.append(
                {
                    "index": idx,
                    "input": item["line"],
                    "matched": True,
                    "school_name": best["院校名称"],
                    "major_name": best["专业名称"],
                    "min_score": serialize_score(best["最低分"]),
                    "min_rank": int(best["最低位次"]),
                    "rank_ratio": round(float(best["位次比"]), 4),
                    "tier": best["分档"],
                    "probability": round(float(best["概率"]), 4),
                    "probability_desc": explain_probability(
                        round(float(best["概率"]), 4),
                        str(best["分档"]),
                        round(float(best["位次比"]), 4),
                    ),
                    "candidate_count": int(len(matched)),
                }
            )

        return {
            "rows": rows,
            "tips": self._build_gradient_tips(rows),
        }

    @staticmethod
    def _build_gradient_tips(rows: List[Dict[str, object]]) -> List[str]:
        valid = [r for r in rows if r.get("matched")]
        if not valid:
            return ["未识别到有效志愿项，暂无法评估梯度。"]

        total = len(valid)
        tier_count = {"冲": 0, "稳": 0, "保": 0}
        for row in valid:
            tier = row.get("tier", "")
            if tier in tier_count:
                tier_count[tier] += 1

        tips: List[str] = []
        if tier_count["保"] == 0:
            tips.append("当前志愿缺少保底项，建议至少增加 1-2 个保底专业。")
        if tier_count["冲"] / total > 0.6:
            tips.append("冲刺项占比偏高，整体风险较大。")
        if tier_count["冲"] == 0 and tier_count["保"] / total >= 0.7:
            tips.append("方案偏保守，可加入少量冲刺项提升上限。")

        probabilities = [float(r["probability"]) for r in valid]
        if total >= 3 and max(probabilities) - min(probabilities) < 0.2:
            tips.append("志愿概率分布较集中，梯度不够明显。")

        first_three = valid[:3]
        if len(first_three) == 3 and all(r["tier"] == "冲" for r in first_three):
            tips.append("前 3 志愿均为冲刺，建议前段加入至少 1 个稳妥项。")

        if not tips:
            tips.append("志愿梯度整体较均衡，可继续按个人偏好微调顺序。")
        return tips
