#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


SAFE_RATIO = 0.80
STEADY_RATIO = 1.10
TIER_DISTRIBUTION = {"冲": 0.25, "稳": 0.45, "保": 0.30}

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

MOCK_SCORE_FILES = {
    "历史": ["历史类联考一分一段表.xlsx", "历史联考一分一段表.xlsx", "hist_mock.xlsx"],
    "物理": ["物理类联考一分一段表.xlsx", "物理联考一分一段表.xlsx", "phys_mock.xlsx"],
}
MOCK_SCORE_DIR = "mock_exams"


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


def resolve_optional_file(base_dir: str, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    return None


def parse_mock_meta_from_filename(filename: str) -> Tuple[str, str]:
    stem = os.path.splitext(filename)[0]
    lower_stem = stem.lower()
    stream = ""
    if ("历史" in stem) or ("hist" in lower_stem) or ("history" in lower_stem):
        stream = "历史"
    elif ("物理" in stem) or ("phys" in lower_stem) or ("physics" in lower_stem):
        stream = "物理"
    if not stream:
        return "", ""

    label = stem
    label = re.sub(r"(历史类?|物理类?)", "", label)
    label = re.sub(r"(?i)(history|hist|physics|phys)", "", label)
    label = re.sub(r"(联考|一分一段表|分数段)", "", label)
    label = re.sub(r"[_\-\s]+", " ", label).strip()
    if not label:
        label = "联考"
    return stream, label


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


def rank_to_score(rank: int, score_map: Dict[float, int]) -> float:
    if rank <= 0:
        rank = 1
    sorted_scores = sorted(score_map.keys(), reverse=True)
    last_score = sorted_scores[-1]
    for s in sorted_scores:
        if int(score_map[s]) >= int(rank):
            return float(s)
    return float(last_score)


def convert_rank_by_percentile(source_rank: int, source_total: int, target_total: int) -> int:
    if source_total <= 0 or target_total <= 0:
        raise ValueError("总人数数据无效，无法进行联考位次换算")
    ratio = float(source_rank) / float(source_total)
    ratio = max(1.0 / float(source_total), min(ratio, 1.0))
    target_rank = int(round(ratio * float(target_total)))
    return max(1, min(target_rank, int(target_total)))


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
    if ratio <= 0.65:
        p = 0.84
    elif ratio <= SAFE_RATIO:
        p = 0.66 + (SAFE_RATIO - ratio) * (0.18 / 0.15)
    elif ratio <= STEADY_RATIO:
        p = 0.40 + (STEADY_RATIO - ratio) * (0.26 / 0.30)
    elif ratio <= 1.30:
        p = 0.20 + (1.30 - ratio) * (0.20 / 0.20)
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

            total_rank = max([int(v) for v in score_map.values()]) if score_map else 0
            mock_tables = self._load_mock_tables(stream)

            ranked_admission = build_ranked_admission(admission_df, score_map, top_score, top_rank)
            self.datasets[stream] = {
                "admission": ranked_admission,
                "score_map": score_map,
                "top_score": top_score,
                "top_rank": top_rank,
                "total_rank": total_rank,
                "mock_tables": mock_tables,
            }

    def _load_mock_tables(self, stream: str) -> Dict[str, Dict[str, object]]:
        if stream not in {"历史", "物理"}:
            return {}

        tables: Dict[str, Dict[str, object]] = {}
        search_dirs = [self.data_dir, os.path.join(self.data_dir, MOCK_SCORE_DIR)]
        reserved = set()
        for paths in STREAM_FILES.values():
            reserved.update(paths.get("admission", []))
            if paths.get("score_table"):
                reserved.add(paths["score_table"])

        candidate_paths: List[str] = []
        for folder in search_dirs:
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                if not file.lower().endswith(".xlsx"):
                    continue
                if file in reserved:
                    continue
                path = os.path.join(folder, file)
                candidate_paths.append(path)

        for legacy_name in MOCK_SCORE_FILES.get(stream, []):
            legacy_path = os.path.join(self.data_dir, legacy_name)
            if os.path.exists(legacy_path):
                candidate_paths.append(legacy_path)

        dedup = []
        seen = set()
        for path in candidate_paths:
            norm = os.path.normpath(path)
            if norm in seen:
                continue
            seen.add(norm)
            dedup.append(path)

        for path in dedup:
            filename = os.path.basename(path)
            stream_from_name, label = parse_mock_meta_from_filename(filename)
            if stream_from_name != stream:
                continue
            try:
                score_map, top_score, top_rank = load_score_table(path)
            except Exception:
                continue
            key = os.path.relpath(path, self.data_dir).replace("\\", "/")
            total_rank = max([int(v) for v in score_map.values()]) if score_map else 0
            tables[key] = {
                "key": key,
                "label": label,
                "filename": filename,
                "path": path,
                "score_map": score_map,
                "top_score": top_score,
                "top_rank": top_rank,
                "total_rank": total_rank,
            }
        return dict(sorted(tables.items(), key=lambda item: item[1]["label"]))

    def get_mock_exam_options(self, stream: str) -> List[Dict[str, str]]:
        stream = normalize_stream(stream)
        if stream not in self.datasets:
            return []
        tables = self.datasets[stream].get("mock_tables", {})
        return [{"key": key, "label": data["label"]} for key, data in tables.items()]

    def _resolve_mock_table(self, stream: str, mock_exam_key: str) -> Dict[str, object]:
        tables = self.datasets[stream].get("mock_tables", {})
        if not tables:
            raise ValueError(
                "未找到联考一分一段表。请在项目目录创建 mock_exams 文件夹，并按 "
                "'历史_考试名.xlsx' 或 '物理_考试名.xlsx' 命名。"
            )
        if mock_exam_key and mock_exam_key in tables:
            return tables[mock_exam_key]
        if mock_exam_key and mock_exam_key not in tables:
            raise ValueError("所选联考数据不存在，请重新选择联考试卷。")
        first_key = next(iter(tables))
        return tables[first_key]

    def rank_from_input(
        self,
        stream: str,
        score: Optional[float],
        rank: Optional[int],
        art_mode: str = "普高",
        art_culture_score: Optional[float] = None,
        art_special_score: Optional[float] = None,
        mock_score: Optional[float] = None,
        mock_rank: Optional[int] = None,
        mock_exam_key: str = "",
    ) -> Tuple[int, Optional[float], Dict[str, object]]:
        stream = normalize_stream(stream)
        if stream not in self.datasets:
            raise ValueError("科类仅支持 历史/物理/艺术")
        used_score: Optional[float] = None
        conversion: Dict[str, object] = {"mode": "direct"}
        data = self.datasets[stream]

        if rank is not None and rank > 0:
            return int(rank), used_score, conversion

        if stream == "艺术":
            if score is not None:
                used_score = float(score)
            else:
                if art_culture_score is None or art_special_score is None:
                    raise ValueError("艺术类请填写综合分，或填写文化成绩与艺考成绩")
                used_score = calc_art_composite(art_mode, float(art_culture_score), float(art_special_score))
            final_rank = score_to_rank(used_score, data["score_map"], data["top_score"], data["top_rank"])
            return final_rank, used_score, conversion
        else:
            if mock_rank is not None or mock_score is not None:
                mock_table = self._resolve_mock_table(stream, mock_exam_key)

                source_rank: Optional[int] = None
                if mock_rank is not None and mock_rank > 0:
                    source_rank = int(mock_rank)
                elif mock_score is not None:
                    source_rank = score_to_rank(
                        float(mock_score),
                        mock_table["score_map"],
                        mock_table["top_score"],
                        mock_table["top_rank"],
                    )
                if source_rank is None:
                    raise ValueError("联考换算需输入联考成绩或联考位次")

                source_total = int(mock_table.get("total_rank") or 0)
                target_total = int(data.get("total_rank") or 0)
                target_rank = convert_rank_by_percentile(source_rank, source_total, target_total)
                predicted_score = rank_to_score(target_rank, data["score_map"])
                final_rank = score_to_rank(predicted_score, data["score_map"], data["top_score"], data["top_rank"])
                used_score = predicted_score
                conversion = {
                    "mode": "mock",
                    "mock_exam_key": mock_table["key"],
                    "mock_exam_label": mock_table["label"],
                    "mock_rank": int(source_rank),
                    "mock_total": int(source_total),
                    "gaokao_rank": int(target_rank),
                    "gaokao_total": int(target_total),
                    "predicted_score": serialize_score(predicted_score),
                }
                return final_rank, used_score, conversion

            if score is not None:
                used_score = float(score)
                final_rank = score_to_rank(used_score, data["score_map"], data["top_score"], data["top_rank"])
                return final_rank, used_score, conversion

            raise ValueError("请至少输入分数、位次或联考成绩")

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
            df = self._select_by_tier_strategy(df, top_n=top_n, per_school_limit=2)

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

    def _select_by_tier_strategy(self, ranked_df: pd.DataFrame, top_n: int, per_school_limit: int) -> pd.DataFrame:
        if ranked_df.empty:
            return ranked_df

        desired = self._build_tier_targets(top_n)
        tier_order = ["冲", "稳", "保"]

        tier_frames: Dict[str, pd.DataFrame] = {}
        for tier in tier_order:
            tier_df = ranked_df[ranked_df["分档"] == tier].sort_values(
                by=["贴合度", "概率", "最低位次"], ascending=[True, False, True]
            )
            tier_frames[tier] = tier_df

        full_sorted = ranked_df.sort_values(by=["贴合度", "概率", "最低位次"], ascending=[True, False, True])
        selected_idx: List[int] = []
        selected_set = set()
        school_count: Dict[str, int] = {}

        def pick_from(frame: pd.DataFrame, need: int, enforce_school_limit: bool = True) -> None:
            if need <= 0:
                return
            for idx, row in frame.iterrows():
                if len(selected_idx) >= top_n:
                    break
                if idx in selected_set:
                    continue
                school = str(row["院校名称"])
                if enforce_school_limit and school_count.get(school, 0) >= per_school_limit:
                    continue
                selected_idx.append(idx)
                selected_set.add(idx)
                school_count[school] = school_count.get(school, 0) + 1
                need -= 1
                if need <= 0:
                    break

        for tier in tier_order:
            pick_from(tier_frames[tier], desired[tier], enforce_school_limit=True)

        if len(selected_idx) < top_n:
            pick_from(full_sorted, top_n - len(selected_idx), enforce_school_limit=True)

        if len(selected_idx) < top_n:
            pick_from(full_sorted, top_n - len(selected_idx), enforce_school_limit=False)

        if not selected_idx:
            return ranked_df.head(top_n)
        return ranked_df.loc[selected_idx]

    @staticmethod
    def _build_tier_targets(top_n: int) -> Dict[str, int]:
        targets = {tier: int(top_n * ratio) for tier, ratio in TIER_DISTRIBUTION.items()}
        remain = top_n - sum(targets.values())
        fill_order = ["稳", "保", "冲"]
        idx = 0
        while remain > 0:
            tier = fill_order[idx % len(fill_order)]
            targets[tier] += 1
            remain -= 1
            idx += 1
        return targets

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
