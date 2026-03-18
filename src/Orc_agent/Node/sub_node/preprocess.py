import os
import re
import io
import json
import warnings

import numpy as np
from collections.abc import Mapping, Sequence
import pandas as pd
from langgraph.graph import END
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import pyarrow.parquet as pq

from ...State.state import preprocessState
from ...core.observe import observe
from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session, merge_runnable_config
from ...core.logger import logger

warnings.filterwarnings("ignore")


# --- Helpers ---

def _fb(state, msg):
    """Return a list of NEW feedback messages (not existing ones).
    merge_logs reducer will handle concatenation with existing state."""
    return [msg]


# 아키텍쳐 변경으로 미사용 처리

# def _df(state):
#     """Get the working DataFrame (cleaned if available, else raw, else load from file_path)."""
#     for key in ["cleaned_dataframe", "raw_dataframe"]:
#         raw = state.get(key)
#         if raw is not None:
#             return raw if isinstance(raw, pd.DataFrame) else pd.read_json(io.StringIO(raw), orient="split")
#     # Fallback: load from file_path (CSV)
#     file_path = state.get("file_path", "")
#     if file_path and os.path.exists(file_path):
#         ext = os.path.splitext(file_path)[1].lower()
#         if ext == ".csv":
#             return pd.read_csv(file_path)
#         elif ext in (".xls", ".xlsx"):
#             return pd.read_excel(file_path)
#         elif ext == ".json":
#             return pd.read_json(file_path)
#     raise ValueError("No DataFrame in state and no valid file_path.")
# def _df_json(state):
#     return _df(state).to_json(orient="split")

def _load_raw_file(state):
    """최초 원본 파일(CSV, Excel 등)의 확장자를 구분하여 읽어옵니다."""
    file_path = state.get("file_path", "")
    if file_path and os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in (".xls", ".xlsx"):
            return pd.read_excel(file_path)
        elif ext == ".json":
            return pd.read_json(file_path)
    
    raise ValueError(f"지원하지 않는 파일 형식 또는 경로 오류: {file_path}")

def _load_parquet(state):
    """state에 저장된 경로에서 DataFrame을 꺼내옵니다."""
    path = state.get("working_df_path")
    if path and os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame() # 에러 방지용

def _save_parquet(df, config, file_name="working_data.parquet"):
    """DataFrame을 임시 폴더에 파케이로 (덮어쓰기) 저장하고 주소만 뱉습니다."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    tmp_dir = os.path.join(os.getcwd(), "tmp", str(thread_id))
    os.makedirs(tmp_dir, exist_ok=True)
    
    save_path = os.path.join(tmp_dir, file_name)
    df.columns = df.columns.astype(str)
    df.to_parquet(save_path, engine='pyarrow', index=False)
    
    return save_path # <--- 반드시 객체가 아닌 '문자열 경로'를 반환

def _clean_for_serialization(obj):
    """모든 넘파이 타입을 파이썬 기본 타입으로 재귀적으로 변환"""
    if isinstance(obj, Mapping):
        return {k: _clean_for_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [_clean_for_serialization(i) for i in obj]
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj): # pd.NA나 np.nan 처리
        return None
    return obj

# --- Tools: Raw Data Preprocessing ---

NUMERIC_KEYWORDS = [
    "impression", "click", "conversion", "conv", "spend", "cost", "budget",
    "revenue", "income", "sales", "profit", "value", "cpc", "cpa", "cpm",
    "ctr", "cvr", "roas", "roi", "aov", "reach", "view", "visit", "session",
    "order", "purchase", "signup", "lead", "install", "ltv", "arpu",
    "amount", "price", "total", "fee", "bid", "rate", "ratio",
]

DATE_KEYWORDS = [
    "date", "time", "day", "week", "month", "year", "period", "quarter",
    "timestamp", "datetime", "created", "start", "end", "at", "updated",
    "launched", "published", "scheduled",
]


@tool
def detect_dirty_numeric_columns(df_path: str) -> list: # 기존: df_json
    """
    Auto-detect columns that SHOULD be numeric but contain formatting artifacts.
    [LLM 지시사항]
    - 인자: df_path (현재 작업 중인 Parquet 파일의 경로 문자열)
    - 역할: $, 쉼표, % 등 특수문자가 포함되어 숫자로 즉시 변환할 수 없는 '더러운 숫자 컬럼'들의 목록을 찾아냅니다.
    """
    #  df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)
    results = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        name_lower = col.lower().strip().replace("_", "").replace(" ", "")
        name_match = any(k in name_lower for k in NUMERIC_KEYWORDS)

        sample = df[col].dropna().astype(str).head(100)
        if len(sample) == 0:
            continue

        cleaned = (sample
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.strip())
        parsed = pd.to_numeric(cleaned, errors="coerce")
        parse_pct = parsed.notna().mean()

        if name_match or parse_pct >= 0.50:
            raw_str = df[col].dropna().astype(str)
            has_dollar = raw_str.str.contains(r"\$", regex=True).any()
            has_comma = raw_str.str.contains(",", regex=False).any()
            has_percent = raw_str.str.contains("%", regex=False).any()
            has_spaces = raw_str.str.contains(r"^\s+|\s+$", regex=True).any()

            results.append({
                "column": col,
                "current_dtype": str(df[col].dtype),
                "name_keyword_match": name_match,
                "parse_success_pct": round(parse_pct * 100, 2),
                "has_dollar_sign": bool(has_dollar),
                "has_commas": bool(has_comma),
                "has_percent": bool(has_percent),
                "has_whitespace": bool(has_spaces),
                "sample_raw": sample.head(3).tolist(),
            })

    return results


@tool
def clean_numeric_column(df_path: str, col: str, fill_na_value: float = 0.0) -> dict:
    """
    Strip $, commas, %, whitespace from a column and convert to float.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로), col (정제할 컬럼명), fill_na_value (결측치 대체값)
    - 역할: 대상 컬럼의 특수기호($, 쉼표, %)를 제거하고 순수 숫자(Float) 타입으로 강제 변환합니다. 실패한 값은 fill_na_value로 채웁니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    if col not in df.columns:
        return {"error": f"Column '{col}' not found."}

    original_dtype = str(df[col].dtype)
    original_nulls = int(df[col].isna().sum())

    cleaned = (df[col].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip())

    numeric = pd.to_numeric(cleaned, errors="coerce")
    coercion_failures = int(numeric.isna().sum()) - original_nulls
    numeric = numeric.astype(float).fillna(fill_na_value)

    return {
        "column": col,
        "original_dtype": original_dtype,
        "final_dtype": "float64",
        "original_nulls": original_nulls,
        "coercion_failures": max(coercion_failures, 0),
        "filled_with": fill_na_value,
        "sample_before": df[col].head(3).tolist(),
        "sample_after": numeric.head(3).tolist(),
    }


@tool
def detect_unparsed_date_columns(df_path: str) -> list:
    """
    Find columns that contain date-like strings but are not yet parsed to datetime.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로)
    - 역할: 날짜/시간 정보를 포함하고 있지만 아직 파이썬 datetime 타입으로 변환되지 않은(문자열 형태인) 컬럼들을 찾아냅니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    results = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        name_match = any(k in col.lower() for k in DATE_KEYWORDS)

        parseable = False
        if pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(30)
            if len(sample) > 0:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    parseable = parsed.notna().mean() > 0.60
                except Exception:
                    pass

        if name_match or parseable:
            results.append({
                "column": col,
                "current_dtype": str(df[col].dtype),
                "name_keyword_match": name_match,
                "parseable": parseable,
                "null_pct": round(df[col].isna().mean() * 100, 2),
                "sample_values": df[col].dropna().head(3).astype(str).tolist(),
            })

    return results


@tool
def coerce_date_column(df_path: str, col: str) -> dict:
    """
    Parse a column to datetime using pd.to_datetime(errors='coerce').
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로), col (변환할 컬럼명)
    - 역할: 문자열로 된 날짜 컬럼을 실제 날짜형(datetime) 객체로 강제 변환합니다. 변환 불가능한 값은 결측치(NaT)로 처리됩니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    if col not in df.columns:
        return {"error": f"Column '{col}' not found."}

    original_dtype = str(df[col].dtype)
    original_nulls = int(df[col].isna().sum())

    parsed = pd.to_datetime(df[col], errors="coerce")
    new_nulls = int(parsed.isna().sum())
    parse_failures = new_nulls - original_nulls

    return {
        "column": col,
        "original_dtype": original_dtype,
        "final_dtype": "datetime64[ns]",
        "original_nulls": original_nulls,
        "parse_failures": max(parse_failures, 0),
        "total_valid_dates": int(parsed.notna().sum()),
        "date_range": [str(parsed.min()), str(parsed.max())] if parsed.notna().any() else None,
    }


# --- Tools: Categorical Standardization ---

BOOL_TRUE_VALS  = {"t", "true", "1", "1.0", "yes", "y", "on", "enabled", "active", "si", "ja"}
BOOL_FALSE_VALS = {"f", "false", "0", "0.0", "no", "n", "off", "disabled", "inactive", "nein"}


@tool
def detect_boolean_like_columns(df_path: str) -> list:
    """
    Scan every column to detect those containing mixed boolean representations.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로)
    - 역할: True/False를 의미하는 다양한 형태(예: Y/N, 1/0, yes/no)가 섞여 있어 불리언(Boolean)으로 통일할 수 있는 컬럼들을 찾습니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    results = []

    for col in df.columns:
        s = df[col].dropna().astype(str).str.strip().str.lower()
        if len(s) == 0:
            continue

        true_mask  = s.isin(BOOL_TRUE_VALS)
        false_mask = s.isin(BOOL_FALSE_VALS)
        bool_pct = (true_mask.sum() + false_mask.sum()) / len(s)

        if bool_pct >= 0.70 and df[col].nunique() <= 15:
            ambiguous = s[~true_mask & ~false_mask]
            results.append({
                "column": col,
                "unique_raw_values": df[col].dropna().unique().tolist()[:20],
                "true_count": int(true_mask.sum()),
                "false_count": int(false_mask.sum()),
                "ambiguous_count": int(len(ambiguous)),
                "ambiguous_values": ambiguous.unique().tolist()[:10],
                "bool_coverage_pct": round(bool_pct * 100, 2),
            })

    return results


@tool
def standardize_boolean_column(df_path: str, col: str) -> dict:
    """
    Unify a column's boolean-like values into canonical True/False.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로), col (통일할 컬럼명)
    - 역할: Y/N, 1/0, True/False 등으로 혼재되어 있는 값을 파이썬 표준 True/False 로 일관성 있게 통일합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    original_nunique = df[col].nunique()

    s = df[col].astype(str).str.strip().str.lower()
    mapping = {}
    for v in s.unique():
        if v in BOOL_TRUE_VALS:
            mapping[v] = True
        elif v in BOOL_FALSE_VALS:
            mapping[v] = False
        elif v == "nan" or v == "" or v == "none":
            mapping[v] = None
        else:
            mapping[v] = None

    converted = s.map(mapping)
    n_unified = original_nunique - converted.nunique()

    return {
        "column": col,
        "original_unique": original_nunique,
        "final_unique": int(converted.nunique()),
        "values_unified": n_unified,
        "mapping_used": {k: str(v) for k, v in mapping.items()},
        "null_after": int(converted.isna().sum()),
    }


@tool
def detect_categorical_synonyms(df_path: str, col: str) -> dict:
    """
    For non-boolean categorical columns, detect likely synonyms.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로), col (검사할 범주형 컬럼명)
    - 역할: 'USA', 'U.S.A', 'usa_' 등 대소문자나 공백/특수문자 차이로 인해 의미가 나뉘어진 동일(동의어) 카테고리들을 탐지합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    if col not in df.columns:
        return {"error": f"Column {col} not found."}

    vals = df[col].dropna().astype(str).unique()
    groups = {}
    for v in vals:
        key = re.sub(r'[\s_\-]+', '', v.strip().lower())
        groups.setdefault(key, []).append(v)

    synonyms = {k: v for k, v in groups.items() if len(v) > 1}

    return {
        "column": col,
        "total_unique": len(vals),
        "synonym_groups": synonyms,
        "groups_found": len(synonyms),
        "values_to_merge": sum(len(v) - 1 for v in synonyms.values()),
    }


@tool
def unify_categorical_synonyms(df_path: str, col: str) -> dict:
    """
    Merge categorical synonyms by normalizing to the most frequent variant.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로), col (통합할 범주형 컬럼명)
    - 역할: 의미가 같은 동의어 쌍(예: 'Apple', 'apple ')들을 가장 많이 등장한 빈도수 중심의 대표 텍스트로 병합(통일)합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    if col not in df.columns:
        return {"error": f"Column {col} not found."}

    s = df[col].dropna().astype(str)
    groups = {}
    for v in s.values:
        key = re.sub(r'[\s_\-]+', '', v.strip().lower())
        groups.setdefault(key, []).append(v)

    mapping = {}
    for key, variants in groups.items():
        counts = pd.Series(variants).value_counts()
        canonical = counts.index[0]
        for v in set(variants):
            if v != canonical:
                mapping[v] = canonical

    return {
        "column": col,
        "mappings_applied": len(mapping),
        "mapping": mapping,
    }


# --- Tools: Duplicate Cleanup ---

@tool
def detect_duplicate_columns(df_path: str) -> dict:
    """
    Detect columns that are likely duplicates based on name similarity or data identity.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로)
    - 역할: 이름이 비슷하거나 내부 데이터가 100% 동일하여 중복 가능성이 높은 컬럼 쌍들을 찾아냅니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    cols = list(df.columns)

    def normalize_name(name):
        return re.sub(r'[\s_\-\.]+', '', str(name).strip().lower())

    name_groups = {}
    for col in cols:
        key = normalize_name(col)
        name_groups.setdefault(key, []).append(col)

    name_dupes = {k: v for k, v in name_groups.items() if len(v) > 1}

    data_dupes = []
    checked = set()
    for i, col_a in enumerate(cols):
        for col_b in cols[i+1:]:
            pair = tuple(sorted([col_a, col_b]))
            if pair in checked:
                continue
            checked.add(pair)
            if normalize_name(col_a) != normalize_name(col_b):
                try:
                    if df[col_a].equals(df[col_b]):
                        data_dupes.append({"columns": [col_a, col_b], "match": "exact_data"})
                    elif df[col_a].astype(str).equals(df[col_b].astype(str)):
                        data_dupes.append({"columns": [col_a, col_b], "match": "string_equal"})
                except Exception:
                    pass

    return {
        "name_based_duplicates": name_dupes,
        "name_groups_found": len(name_dupes),
        "data_based_duplicates": data_dupes,
        "data_dupes_found": len(data_dupes),
        "total_columns_before": len(cols),
    }


@tool
def detect_duplicate_rows(df_path: str) -> dict:
    """
    Detect fully duplicate rows and near-duplicate rows.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로)
    - 역할: 모든 컬럼 값이 완전히 일치하는 중복 행의 개수나, 공백 무시 시 중복되는 행을 세어 알려줍니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    exact_dupes = int(df.duplicated().sum())

    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=["object", "string"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    near_dupes = int(df_clean.duplicated().sum()) - exact_dupes

    return {
        "total_rows": len(df),
        "exact_duplicate_rows": exact_dupes,
        "near_duplicate_rows": near_dupes,
        "total_removable": exact_dupes + near_dupes,
        "pct_removable": round((exact_dupes + near_dupes) / len(df) * 100, 2) if len(df) > 0 else 0,
    }


@tool
def detect_blank_columns(df_path: str) -> list:
    """
    Detect columns that are entirely blank, all-NaN, or all-zero.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로)
    - 역할: 전체가 NaN이거나 빈 문자열, 0으로만 채워진 '의미 없는 빈 컬럼' 목록을 찾아냅니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    blanks = []
    for col in df.columns:
        s = df[col]
        is_blank = False
        if s.isna().all():
            is_blank = True
            reason = "all NaN"
        elif s.astype(str).str.strip().replace("", pd.NA).isna().all():
            is_blank = True
            reason = "all empty strings"
        elif pd.api.types.is_numeric_dtype(s) and (s.fillna(0) == 0).all():
            is_blank = True
            reason = "all zeros"

        if is_blank:
            blanks.append({"column": col, "reason": reason})

    return blanks


# --- Tools: Date Integrity ---

DATE_COL_KEYWORDS = ["date", "time", "day", "week", "month", "period", "timestamp",
                     "datetime", "created", "start", "end", "at"]


@tool
def detect_date_columns(df_path: str) -> dict:
    """
    Identify which columns contain date/time information.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로)
    - 역할: 전체 컬럼 중 날짜/시간 정보를 담고 있는 시간 차원(Temporal Dimension) 컬럼들이 무엇인지 자동 판단합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    date_cols = []
    for col in df.columns:
        name_match = any(k in col.lower() for k in DATE_COL_KEYWORDS)
        type_match = pd.api.types.is_datetime64_any_dtype(df[col])
        parseable = False
        if not type_match and pd.api.types.is_string_dtype(df[col]):
            try:
                parsed = pd.to_datetime(df[col].dropna().head(20), errors="coerce")
                parseable = parsed.notna().mean() > 0.8
            except Exception:
                pass

        if name_match or type_match or parseable:
            date_cols.append({
                "column": col,
                "name_match": name_match,
                "dtype_match": type_match,
                "parseable": parseable,
                "null_pct": round(df[col].isna().mean() * 100, 2),
                "sample_values": df[col].dropna().head(3).astype(str).tolist(),
            })

    return {
        "date_columns": date_cols,
        "count": len(date_cols),
        "has_any_date": len(date_cols) > 0,
    }


@tool
def analyze_date_completeness(df_path: str, date_col: str, campaign_col: str = "") -> dict:
    """
    Analyze missing dates and decide on strategy.
    [LLM 지시사항]
    - 인자: df_path(Parquet 경로), date_col(날짜 컬럼), campaign_col(선택사항, 캠페인 식별용 컬럼)
    - 역할: 날짜 컬럼의 결측치(비어있는 곳) 비율을 조사하고, 제거할지 또는 이전 캠페인 날짜로 채울지에 대한 데이터 복원/삭제 전략을 추천합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    if date_col not in df.columns:
        return {"error": f"Date column '{date_col}' not found."}

    total = len(df)
    missing_mask = df[date_col].isna()
    n_missing = int(missing_mask.sum())
    pct_missing = round(n_missing / total * 100, 2) if total > 0 else 0

    strategy = "none"
    fill_map = {}

    if n_missing == 0:
        strategy = "none_needed"
    elif pct_missing <= 5.0:
        strategy = "drop_missing"
    elif campaign_col and campaign_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            earliest = df.groupby(campaign_col)[date_col].min()
            n_fillable = 0
            for camp, earliest_date in earliest.items():
                if pd.notna(earliest_date):
                    camp_missing = (df[campaign_col] == camp) & missing_mask
                    n_fillable += int(camp_missing.sum())
                    if camp_missing.any():
                        fill_map[str(camp)] = str(earliest_date)

            if n_fillable > 0:
                strategy = "fill_from_campaign_earliest"
            else:
                strategy = "drop_missing"
        except Exception:
            strategy = "drop_missing"
    else:
        strategy = "drop_missing" if pct_missing <= 20 else "flag_for_review"

    return {
        "date_column": date_col,
        "total_rows": total,
        "missing_dates": n_missing,
        "missing_pct": pct_missing,
        "strategy": strategy,
        "fill_map": fill_map,
        "fillable_rows": len(fill_map),
    }


# --- Tools: Analysis & Measurement ---

@tool
def profile_all_columns(df_path: str) -> dict:
    """
    Profile every column: dtype, nulls, unique count, range, sparsity, skew.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로)
    - 역할: 각 컬럼별 고유값 개수, 결측치 비율, 빈도 등을 망라한 '메타데이터 프로필 요약본'을 제공하여 전체 데이터 상태 파악을 돕습니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    profiles = {}
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        profiles[col] = {
            "dtype": str(s.dtype), "null_pct": round(s.isna().mean() * 100, 2),
            "nunique": int(s.nunique()),
            "min": float(s.min()) if is_num else None,
            "max": float(s.max()) if is_num else None,
            "mean": round(float(s.mean()), 4) if is_num else None,
            "std": round(float(s.std()), 4) if is_num else None,
            "zero_pct": round((s == 0).mean() * 100, 2) if is_num else None,
            "skew": round(float(s.skew()), 4) if is_num else None,
        }
    return profiles


@tool
def detect_existing_metrics(df_path: str) -> dict:
    """
    Detect which derived metrics already exist in the data.
    [LLM 지시사항]
    - 인자: df_path (Parquet 경로)
    - 역할: CTR, CVR, ROAS 등 파생 지표가 이미 컬럼으로 계산되어 데이터에 존재하는지 여부를 자동 감지하여 알려줍니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    cols_lower = {c: c.lower() for c in df.columns}
    known = {
        "CTR": ["ctr", "click_through_rate"], "CVR": ["cvr", "conversion_rate"],
        "CPC": ["cpc", "cost_per_click"], "CPA": ["cpa", "cost_per_acquisition", "cost_per_conversion"],
        "ROAS": ["roas", "return_on_ad_spend"], "CPM": ["cpm", "cost_per_mille"],
        "AOV": ["aov", "average_order_value"],
    }
    found = {}
    for name, aliases in known.items():
        for orig, low in cols_lower.items():
            if any(a in low for a in aliases):
                found[name] = orig; break
    return {"existing_metrics": found, "total_columns": len(df.columns)}


@tool
def classify_column_role(col_name: str, profile: dict) -> dict:
    """
    Classify column into: volume, performance, efficiency, value, segment, time, identifier.
    [LLM 지시사항]
    - 인자: col_name(대상 컬럼명), profile(해당 컬럼의 프로필 딕셔너리)
    - 역할: 이 컬럼이 비용, 수익, 트래픽, 식별자 중 마케팅적으로 어떤 역할을 담당하는지 의미적 계층 분류를 생성합니다.
    """
    name = col_name.lower()
    rules = [
        (["id", "key", "hash", "uuid"], "identifier", "Identifier column."),
        (["date", "time", "day", "week", "month", "year", "period", "quarter"], "time_dimension", "Temporal dimension."),
        (["campaign", "channel", "source", "medium", "device", "geo", "country", "region",
          "segment", "ad_group", "adgroup", "ad_set", "platform", "network", "placement",
          "audience", "objective", "type", "category", "brand"], "segment_dimension", "Segment dimension."),
        (["impression", "reach", "view", "session", "visit"], "volume_exposure", "Exposure volume."),
        (["click"], "volume_engagement", "Engagement volume."),
        (["conversion", "conv", "purchase", "order", "signup", "lead", "install"], "volume_outcome", "Outcome volume."),
        (["cost", "spend", "budget"], "cost_metric", "Cost signal."),
        (["revenue", "income", "sales", "value", "profit", "ltv", "arpu"], "value_metric", "Value signal."),
        (["ctr", "cvr", "rate", "ratio", "pct"], "performance_rate", "Existing rate."),
        (["cpc", "cpa", "cpm", "roas", "roi", "aov"], "efficiency_metric", "Existing efficiency."),
    ]
    for keywords, role, just in rules:
        if any(k in name for k in keywords):
            return {"column": col_name, "role": role, "justification": just}
    if profile.get("nunique", 999) < 30 and str(profile.get("dtype", "")).startswith(("object", "cat", "str", "String")):
        return {"column": col_name, "role": "segment_dimension", "justification": "Low cardinality."}
    zero_pct = profile.get("zero_pct")
    if zero_pct is not None and zero_pct > 70:
        return {"column": col_name, "role": "noise_artifact", "justification": ">70% zeros."}
    return {"column": col_name, "role": "unclassified", "justification": "Unclassified."}


@tool
def derive_rate_metric(df_path: str, numerator_col: str, denominator_col: str,
                       metric_name: str, business_question: str) -> dict:
    """
    Derive a rate metric. No smoothing. Real values only.
    [LLM 지시사항]
    - 인자: df_path(경로), numerator_col(분자), denominator_col(분모), metric_name(지표명), business_question(목적)
    - 역할: CTR(클릭/노출) 등 두 컬럼의 나누기 비율을 검증해 파생 가능한 비율 지표의 통계를 계산합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    num, den = df[numerator_col].fillna(0), df[denominator_col].fillna(0)
    rate = np.where(den > 0, num / den, np.nan)
    valid = int(np.sum(~np.isnan(rate)))
    return {"metric_name": metric_name, "formula": f"{numerator_col} / {denominator_col}",
            "business_question": business_question, "valid_rows": valid,
            "zero_denom_rows": int(np.sum(den == 0)),
            "mean": round(float(np.nanmean(rate)), 6) if valid > 0 else None}


@tool
def derive_efficiency_metric(df_path: str, cost_col: str, event_col: str,
                             metric_name: str, business_question: str) -> dict:
    """
    Derive cost-efficiency metric. Real values, no transforms.
    [LLM 지시사항]
    - 인자: df_path(경로), cost_col(비용), event_col(사건), metric_name(지표명), business_question(목적)
    - 역할: CPC(비용/클릭), CPA(비용/획득) 등 단위 성과당 소모 비용을 나타내는 효율성 지표를 계산하고 유효성을 검증합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    cost, events = df[cost_col].fillna(0), df[event_col].fillna(0)
    metric = np.where(events > 0, cost / events, np.nan)
    valid = int(np.sum(~np.isnan(metric)))
    return {"metric_name": metric_name, "formula": f"{cost_col} / {event_col}",
            "business_question": business_question, "valid_rows": valid,
            "zero_denom_rows": int(np.sum(events == 0)),
            "mean": round(float(np.nanmean(metric)), 4) if valid > 0 else None}


@tool
def add_reliability_flags(df_path: str, denominator_col: str, threshold: int = 30) -> dict:
    """
    Flag rows where denominator is too small for a stable ratio.
    [LLM 지시사항]
    - 인자: df_path(경로), denominator_col(분모 컬럼), threshold(최소기준, 기본 30)
    - 역할: 기준 표본수(예: 노출 수 30 미만)가 너무 적어 전환율 계산 등에 신뢰할 수 없는 데이터 행 개수를 산출합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    if denominator_col not in df.columns:
        return {"error": f"{denominator_col} not found."}
    return {"denominator": denominator_col,
            "low_volume_count": int((df[denominator_col] < threshold).sum()),
            "low_volume_pct": round((df[denominator_col] < threshold).mean() * 100, 2),
            "zero_division_count": int((df[denominator_col] == 0).sum()),
            "missing_count": int(df[denominator_col].isna().sum()),
            "threshold_used": threshold}


@tool
def detect_anomalies(df_path: str, roles: dict) -> list:
    """
    Detect: extreme ROAS, high CTR + low conv, high spend + zero revenue.
    [LLM 지시사항]
    - 인자: df_path(Parquet 경로), roles(컬럼별 역할 딕셔너리)
    - 역할: 비용은 높은데 수익이 0이거나, 클릭이 지나치게 높지만 전환이 없는 등 마케팅 논리상 비정상(Anomaly)적 패턴을 찾아냅니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    anomalies = []
    cost_cols = [c for c, r in roles.items() if r == "cost_metric" and c in df.columns]
    rev_cols = [c for c, r in roles.items() if r == "value_metric" and c in df.columns]
    vol_out = [c for c, r in roles.items() if r == "volume_outcome" and c in df.columns]
    vol_eng = [c for c, r in roles.items() if r == "volume_engagement" and c in df.columns]
    for cc in cost_cols:
        for rc in rev_cols:
            mask = (df[cc] > df[cc].quantile(0.75)) & (df[rc] == 0)
            if mask.any():
                anomalies.append({"type": "high_spend_zero_revenue", "columns": [cc, rc],
                                  "affected_rows": int(mask.sum()),
                                  "description": f"{mask.sum()} rows: high {cc}, zero {rc}."})
    for ec in vol_eng:
        for oc in vol_out:
            mask = (df[ec] > df[ec].quantile(0.75)) & (df[oc] == 0)
            if mask.any():
                anomalies.append({"type": "high_engagement_zero_outcome", "columns": [ec, oc],
                                  "affected_rows": int(mask.sum()),
                                  "description": f"{mask.sum()} rows: high {ec}, zero {oc}."})
    return anomalies


@tool
def compute_funnel_leakage(df_path: str, funnel_cols: list) -> dict:
    """
    Compute loss rate at each funnel stage.
    [LLM 지시사항]
    - 인자: df_path(Parquet 경로), funnel_cols(단계별 퍼널 컬럼명 리스트)
    - 역할: 퍼널(A->B->C 단계) 별로 이탈률/전환율을 계산하여 유저가 가장 많이 빠져나가는 병목 지점을 수치화합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    leakage = {}
    for i in range(len(funnel_cols) - 1):
        p, c = funnel_cols[i], funnel_cols[i + 1]
        if p in df.columns and c in df.columns:
            ps, cs = df[p].sum(), df[c].sum()
            rate = 1 - (cs / ps) if ps > 0 else None
            leakage[f"{p}_to_{c}"] = {"parent_total": int(ps), "child_total": int(cs),
                                       "loss_pct": round(rate * 100, 2) if rate is not None else None}
    if leakage:
        worst = max(leakage.items(), key=lambda x: x[1].get("loss_pct", 0) or 0)
        leakage["_worst_stage"] = worst[0]
    return leakage


@tool
def compute_share_metrics(df_path: str, value_col: str, segment_col: str = "") -> dict:
    """
    Compute contribution share: segment_value / total_value.
    [LLM 지시사항]
    - 인자: df_path(Parquet 경로), value_col(계산할 값 컬럼), segment_col(구분 기준 컬럼)
    - 역할: 특정 세그먼트나 값이 전체 총합에서 차지하는 비중(기여도 %)을 분석합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    total = df[value_col].sum()
    if total == 0:
        return {"error": f"Total {value_col} is zero."}
    return {"metric": f"{value_col}_share", "formula": f"{value_col} / total_{value_col}",
            "total": round(float(total), 2)}


@tool
def compute_period_change(df_path: str, value_col: str, time_col: str) -> dict:
    """
    Compute period-over-period % change.
    [LLM 지시사항]
    - 인자: df_path(Parquet 경로), value_col(지표 컬럼), time_col(흐름을 나타내는 시간 컬럼)
    - 역할: 이전 기간 대비 현재 기간의 측정 지표 변화율(MoM, WoW 등 증감 퍼센트)을 산출해 트렌드를 계산합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)

    if time_col not in df.columns or value_col not in df.columns:
        return {"error": f"Missing {time_col} or {value_col}."}
    grouped = df.sort_values(time_col).groupby(time_col)[value_col].sum()
    pct_change = grouped.pct_change().dropna()
    return {"metric": f"{value_col}_change_rate", "periods": len(grouped),
            "mean_change_pct": round(float(pct_change.mean()) * 100, 2) if len(pct_change) > 0 else None,
            "last_change_pct": round(float(pct_change.iloc[-1]) * 100, 2) if len(pct_change) > 0 else None}


@tool
def detect_saturation(df_path: str, cost_col: str, conversion_col: str, impression_col: str) -> dict:
    """
    Detect diminishing returns: cost up but conversion rate down across spend quantiles.
    [LLM 지시사항]
    - 인자: df_path(Parquet 경로), cost_col(비용), conversion_col(전환), impression_col(노출)
    - 역할: 지출이 증가함에 따라 효율이 오히려 감소하는 한계 수확 체감(Saturation) 병목 지점을 사분위수로 탐지합니다.
    """
    # df = pd.read_json(io.StringIO(df_json), orient="split")
    df = pd.read_parquet(df_path)
    
    try:
        df["_q"] = pd.qcut(df[cost_col], q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        grouped = df.groupby("_q", observed=True).apply(
            lambda g: g[conversion_col].sum() / g[impression_col].sum() if g[impression_col].sum() > 0 else 0)
        vals = grouped.values.tolist()
        return {"cost_col": cost_col,
                "quartile_rates": {str(k): round(v, 6) for k, v in grouped.items()},
                "diminishing_return_detected": len(vals) >= 3 and vals[-1] < vals[0]}
    except Exception as e:
        return {"error": str(e)}


# --- Node 0: Intake ---

@observe(name="Intake")
def intake_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    # df = _df(state)
    # fb = _fb(state, f"[INTAKE] Received {df.shape[0]:,} rows x {df.shape[1]} cols.")
    # fb.append(f"[INTAKE] Columns: {list(df.columns)}")
    # logger.info(f">>> [Intake] {df.shape[0]} rows x {df.shape[1]} cols")
    # return {"raw_dataframe": df,
    #         "current_stage": "raw_data_preprocessing",
    #         "iteration_count": 0, "error": None,
    #         "agent_feedback": fb, "steps_log": fb}

    df = _load_raw_file(state)

    # 작업: 객체 대신 경로 만들기 
    raw_path = _save_parquet(df, config, "raw_data.parquet")
    working_path = _save_parquet(df, config, "working_data.parquet")

    fb = _fb(state, f"[INTAKE] Received {df.shape[0]:,} rows x {df.shape[1]} cols.")
    fb.append(f"[INTAKE] Columns: {list(df.columns)}")
    logger.info(f">>> [Intake] {df.shape[0]} rows x {df.shape[1]} cols")

    return {
        "raw_df_path": raw_path,
        "working_df_path": working_path,
        "current_stage": "raw_data_preprocessing",
        "iteration_count": 0, "error": None,
        # "agent_feedback": fb,     # 중복으로 데이터 저장하고 있어 제거
        "steps_log": fb
    }


# --- Node 1: Raw Data Preprocessing ---

@observe(name="RawDataPreprocessing")
def raw_data_preprocessing_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    """
    First-pass cleaning of raw messy data:
      1. Auto-detect numeric columns containing $, commas, %, whitespace
      2. Strip those artifacts and convert to float64
      3. Fill NaN in numeric columns with 0 (safe for division later)
      4. Auto-detect and parse date columns to datetime
    """
    
    # df = _df(state).copy()
    # df_j = df.to_json(orient="split")

    # 헬퍼 함수로 이전 노드(Intake)가 저장한 파케이 파일 불러오기
    df = _load_parquet(state)
    # 잠시 현재 상태를 파일로 저장해서 Tool들에게 주소 알려주기
    current_path = _save_parquet(df, config, "temp_working.parquet")

    report = {
        "numeric_columns_cleaned": [],
        "date_columns_parsed": [],
        "columns_before": list(df.columns),
        "dtypes_before": {col: str(df[col].dtype) for col in df.columns},
    }
    fb = []

    # Step 1: Detect and clean dirty numeric columns
    # Tool에게 무거운 df_j 텍스트 대신 깔끔한 주소(current_path)만 전달
    dirty_nums = detect_dirty_numeric_columns.invoke({"df_path": current_path}) # "df_json": df_j → "df_path": current_path로 변경

    fb.append(f"[RAW-PREPROCESS] Found {len(dirty_nums)} dirty numeric column(s).")
    logger.info(f"[RAW-PREPROCESS] Found {len(dirty_nums)} dirty numeric column(s).")

    for info in dirty_nums:
        col = info["column"]
        if col not in df.columns:
            continue

        artifacts = []
        if info.get("has_dollar_sign"): artifacts.append("$")
        if info.get("has_commas"): artifacts.append(",")
        if info.get("has_percent"): artifacts.append("%")
        if info.get("has_whitespace"): artifacts.append("whitespace")

        clean_result = clean_numeric_column.invoke({
            "df_path": current_path, "col": col, "fill_na_value": 0.0  # "df_json": df_j → "df_path": current_path로 변경
        })

        cleaned_series = (df[col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip())
        df[col] = pd.to_numeric(cleaned_series, errors="coerce").astype(float).fillna(0.0)

        report["numeric_columns_cleaned"].append({
            "column": col,
            "artifacts_found": artifacts,
            "original_dtype": clean_result.get("original_dtype", "?"),
            "coercion_failures": clean_result.get("coercion_failures", 0),
        })
        fb.append(f"[RAW-PREPROCESS] Cleaned numeric: '{col}' "
                  f"(removed {artifacts}, {clean_result.get('coercion_failures', 0)} unparseable -> NaN -> 0)")

    # Step 1b: Fill NaN in already-numeric columns
    for col in df.select_dtypes(include=["number"]).columns:
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            df[col] = df[col].fillna(0.0)
            fb.append(f"[RAW-PREPROCESS] Filled {n_nan} NaN with 0 in already-numeric '{col}'.")

    # Step 2: Detect and parse date columns
    # df_j2 = df.to_json(orient="split")

    # df가 수정되었으니 새로 저장해서 최신 주소 만들기
    current_path = _save_parquet(df, config, "temp_working.parquet")
    unparsed_dates = detect_unparsed_date_columns.invoke({"df_path": current_path})  # "df_json": df_j2 → "df_path": current_path로 변경

    fb.append(f"[RAW-PREPROCESS] Found {len(unparsed_dates)} unparsed date column(s).")
    
    for info in unparsed_dates:
        col = info["column"]
        if col not in df.columns:
            continue

        coerce_result = coerce_date_column.invoke({"df_path": current_path, "col": col})  # "df_json": df_j2 → "df_path": current_path로 변경

        df[col] = pd.to_datetime(df[col], errors="coerce")

        report["date_columns_parsed"].append({
            "column": col,
            "original_dtype": coerce_result.get("original_dtype", "?"),
            "parse_failures": coerce_result.get("parse_failures", 0),
            "valid_dates": coerce_result.get("total_valid_dates", 0),
            "date_range": coerce_result.get("date_range"),
        })
        fb.append(f"[RAW-PREPROCESS] Parsed date: '{col}' "
                  f"({coerce_result.get('total_valid_dates', 0)} valid, "
                  f"{coerce_result.get('parse_failures', 0)} failures)")

    report["dtypes_after"] = {col: str(df[col].dtype) for col in df.columns}

    n_changes = len(report["numeric_columns_cleaned"]) + len(report["date_columns_parsed"])
    fb.append(f"[RAW-PREPROCESS] Complete: {n_changes} column(s) cleaned/parsed.")
    logger.info(f">>> [RawDataPreprocessing] {n_changes} column(s) cleaned/parsed.")

    final_path = _save_parquet(df, config, "working_data.parquet")

    result_data = {
        # "cleaned_dataframe": df,
        "working_df_path": final_path, # Return 값에서 무거운 객체(df) 제거하고 주소만 전달
        "raw_preprocessing_report": report,
        "current_stage": "categorical_standardization",
        # "agent_feedback": fb,
        "steps_log": fb,
    }
    return _clean_for_serialization(result_data)


# --- Node 2: Categorical Standardization ---

@observe(name="CategoricalStandardization")
def categorical_standardization_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    """
    Unify mixed boolean representations:
      T, True, 1, Yes, Y  ->  True
      F, False, 0, No, N  ->  False
    Also unify case/whitespace variants in other categoricals.
    """
    # df = _df(state).copy()
    # df_j = df.to_json(orient="split")

    # 전 노드에서 릴레이로 건네준 '최신 파케이 주소(working_df_path)'를 열어서 데이터프레임 꺼내기
    df = _load_parquet(state)
    # Tool에게 데이터를 검사시키기 전에, 방금 최신화된 df를 임시 파일로 굽고 그 주소 받기
    current_path = _save_parquet(df, config, "temp_working.parquet")

    report = {"boolean_columns_fixed": [], "categorical_columns_unified": []}
    fb = []

    # -- Boolean standardization --
    bool_cols = detect_boolean_like_columns.invoke({"df_path": current_path})  # "df_json": df_j → "df_path": current_path로 변경
    for info in bool_cols:
        col = info["column"]
        result = standardize_boolean_column.invoke({"df_path": current_path, "col": col})  # "df_json": df_j → "df_path": current_path로 변경

        s = df[col].astype(str).str.strip().str.lower()
        df[col] = s.map(lambda v: True if v in BOOL_TRUE_VALS
                         else (False if v in BOOL_FALSE_VALS else None))
        report["boolean_columns_fixed"].append(result)
        fb.append(f"[CAT-STD] Boolean unified: '{col}' "
                  f"({info['true_count']} true, {info['false_count']} false, "
                  f"{info['ambiguous_count']} ambiguous -> NaN)")

    current_path = _save_parquet(df, config, "temp_working.parquet")

    # -- Non-boolean categorical synonym unification --
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        if col in [b["column"] for b in bool_cols]:
            continue
        if df[col].nunique() > 200:
            continue

        syn_result = detect_categorical_synonyms.invoke({"df_path": current_path, "col": col})  # "df_json": df_j → "df_path": current_path로 변경
        if syn_result.get("groups_found", 0) > 0:
            unify = unify_categorical_synonyms.invoke({"df_path": current_path, "col": col})
            if unify.get("mappings_applied", 0) > 0:
                mapping = unify["mapping"]
                df[col] = df[col].replace(mapping)
                report["categorical_columns_unified"].append({
                    "column": col,
                    "groups_merged": syn_result["groups_found"],
                    "values_merged": unify["mappings_applied"],
                    "sample_mapping": dict(list(mapping.items())[:5]),
                })
                fb.append(f"[CAT-STD] Synonyms unified: '{col}' "
                          f"({unify['mappings_applied']} value variants merged)")

    n_fixed = len(report["boolean_columns_fixed"]) + len(report["categorical_columns_unified"])
    fb.append(f"[CAT-STD] Total: {n_fixed} columns standardized.")
    logger.info(f">>> [CategoricalStd] {n_fixed} columns standardized.")

    final_path = _save_parquet(df, config, "working_data.parquet")

    result_data = {
        # "cleaned_dataframe": df,
        "working_df_path": final_path,
        "categorical_standardization_report": report,
        "current_stage": "duplicate_cleanup",
        # "agent_feedback": fb,
        "steps_log": fb,
    }
    return _clean_for_serialization(result_data)


# --- Node 3: Duplicate Cleanup ---

@observe(name="DuplicateCleanup")
def duplicate_cleanup_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    """
    1. Merge columns with same name (ignoring spaces/underscores).
    2. Remove blank (all-NaN/all-zero) columns.
    3. Remove exact and near-duplicate rows.
    """
    # df = _df(state).copy()
    # df_j = df.to_json(orient="split")

    df = _load_parquet(state)
    current_path = _save_parquet(df, config, "temp_working.parquet")

    report = {"columns_merged": [], "blank_columns_dropped": [],
              "rows_before": len(df), "rows_after": 0,
              "exact_dupes_removed": 0, "near_dupes_removed": 0}
    fb = []

    # -- Step 1: Merge duplicate-named columns --
    col_dupes = detect_duplicate_columns.invoke({"df_path": current_path})  # "df_json": df_j → "df_path": current_path로 변경
    name_groups = col_dupes.get("name_based_duplicates", {})

    for norm_key, group_cols in name_groups.items():
        if len(group_cols) < 2:
            continue

        def name_quality(n):
            stripped = n.strip().strip("_").strip()
            return (len(stripped), stripped)

        canonical = sorted(group_cols, key=name_quality)[0].strip().strip("_").strip()
        others = [c for c in group_cols if c != canonical]

        for other in others:
            if other in df.columns and canonical in df.columns:
                df[canonical] = df[canonical].fillna(df[other])
                df.drop(columns=[other], inplace=True)
                report["columns_merged"].append(
                    {"kept": canonical, "merged_from": other, "norm_key": norm_key})
                fb.append(f"[DEDUP-COL] Merged '{other}' -> '{canonical}'")
            elif other in df.columns:
                df.rename(columns={other: canonical}, inplace=True)
                fb.append(f"[DEDUP-COL] Renamed '{other}' -> '{canonical}'")

    # Also handle data-identical columns with different names
    if len(df.columns) > 1:
        # df_j2 = df.to_json(orient="split")
        current_path = _save_parquet(df, config, "temp_working.parquet")
        data_dupes = detect_duplicate_columns.invoke({"df_path": current_path}).get("data_based_duplicates", [])  # "df_json": df_j2 → "df_path": current_path로 변경

        for dd in data_dupes:
            cols = dd["columns"]
            if all(c in df.columns for c in cols):
                keep, drop = cols[0], cols[1]
                df.drop(columns=[drop], inplace=True)
                report["columns_merged"].append(
                    {"kept": keep, "merged_from": drop, "reason": "identical_data"})
                fb.append(f"[DEDUP-COL] Dropped data-identical '{drop}' (same as '{keep}')")


    # -- Step 2: Drop blank columns --
    # df_j3 = df.to_json(orient="split")
    current_path = _save_parquet(df, config, "temp_working.parquet")
    blanks = detect_blank_columns.invoke({"df_path": current_path})  # "df_json": df_j3 → "df_path": current_path로 변경

    for b in blanks:
        col = b["column"]
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            report["blank_columns_dropped"].append(b)
            fb.append(f"[DEDUP-COL] Dropped blank column '{col}' ({b['reason']})")

    # -- Step 3: Remove duplicate rows --
    n_before = len(df)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    n_exact = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)

    report["exact_dupes_removed"] = n_exact
    report["rows_after"] = n_after
    fb.append(f"[DEDUP-ROW] {n_before} -> {n_after} rows ({n_exact} duplicates removed).")
    fb.append(f"[DEDUP] Summary: {len(report['columns_merged'])} cols merged, "
              f"{len(report['blank_columns_dropped'])} blanks dropped, "
              f"{n_exact} row dupes removed.")
    logger.info(f">>> [DuplicateCleanup] {n_before}->{n_after} rows, "
                f"{len(report['columns_merged'])} cols merged")

    final_path = _save_parquet(df, config, "working_data.parquet")

    result_data = {
        # "cleaned_dataframe": df,
        "working_df_path": final_path,
        "duplicate_cleanup_report": report,
        "current_stage": "date_integrity",
        # "agent_feedback": fb,
        "steps_log": fb,
    }
    return _clean_for_serialization(result_data)


# --- Node 4: Date Integrity ---

@observe(name="DateIntegrity")
def date_integrity_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    """
    Ensure every event row has a valid date.
    Strategy:
      - <=5% missing dates:  drop those rows
      - >5% with campaign:   fill from earliest campaign date
      - >5% without campaign: flag for review
    """
    # df = _df(state).copy()
    # df_j = df.to_json(orient="split")

    df = _load_parquet(state)
    current_path = _save_parquet(df, config, "temp_working.parquet")

    report = {"date_columns_found": [], "actions_taken": [], "rows_before": len(df), "rows_after": 0}
    fb = []

    # Detect date columns
    date_info = detect_date_columns.invoke({"df_path": current_path})  # "df_json": df_j → "df_path": current_path로 변경
    report["date_columns_found"] = date_info["date_columns"]

    if not date_info["has_any_date"]:
        fb.append("[DATE] WARNING: No date columns detected. Date integrity cannot be enforced.")
        fb.append("[DATE] Events without dates cannot be assessed for temporal performance.")
        report["actions_taken"].append({"action": "no_date_column_found", "note": "Cannot enforce date integrity."})
        report["rows_after"] = len(df)
        logger.info(">>> [DateIntegrity] No date columns found — skipped.")

        final_path = _save_parquet(df, config, "working_data.parquet")

        result_data = {
            # "cleaned_dataframe": df,
            "working_df_path": final_path,
            "date_integrity_report": report,
            "current_stage": "data_state_awareness",
            # "agent_feedback": fb,
            "steps_log": fb,
        }
        return _clean_for_serialization(result_data)

    # Find campaign/segment columns to use for backfill
    campaign_col = ""
    for col in df.columns:
        if any(k in col.lower() for k in ["campaign", "ad_group", "adgroup", "channel"]):
            campaign_col = col
            break

    # Process each date column
    for dc_info in date_info["date_columns"]:
        date_col = dc_info["column"]

        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            fb.append(f"[DATE] Could not parse '{date_col}' as datetime. Skipping.")
            continue

        current_path = _save_parquet(df, config, "temp_working.parquet")
        analysis = analyze_date_completeness.invoke({
            "df_path": current_path, # "df_json": df.to_json(orient="split") → "df_path": current_path로 변경
            "date_col": date_col,
            "campaign_col": campaign_col,
        })

        strategy = analysis.get("strategy", "none_needed")
        missing_n = analysis.get("missing_dates", 0)
        missing_pct = analysis.get("missing_pct", 0)

        fb.append(f"[DATE] Column '{date_col}': {missing_n} missing ({missing_pct}%) -> strategy: {strategy}")

        if strategy == "none_needed":
            report["actions_taken"].append({
                "column": date_col, "action": "none", "reason": "No missing dates."})

        elif strategy == "drop_missing":
            before = len(df)
            df = df[df[date_col].notna()].reset_index(drop=True)
            dropped = before - len(df)
            report["actions_taken"].append({
                "column": date_col, "action": "dropped_rows",
                "rows_dropped": dropped, "reason": f"{missing_pct}% missing (<=5% threshold)."})
            fb.append(f"[DATE] Dropped {dropped} rows with missing '{date_col}'.")

        elif strategy == "fill_from_campaign_earliest":
            fill_map = analysis.get("fill_map", {})
            filled = 0
            for camp, earliest in fill_map.items():
                mask = (df[campaign_col] == camp) & df[date_col].isna()
                df.loc[mask, date_col] = pd.to_datetime(earliest)
                filled += int(mask.sum())

            still_missing = df[date_col].isna().sum()
            if still_missing > 0:
                df = df[df[date_col].notna()].reset_index(drop=True)

            report["actions_taken"].append({
                "column": date_col, "action": "filled_from_campaign",
                "rows_filled": filled, "rows_dropped_remaining": int(still_missing),
                "fill_map": fill_map,
                "reason": f"Filled from earliest date in '{campaign_col}'."})
            fb.append(f"[DATE] Filled {filled} missing dates from campaign earliest. "
                      f"Dropped {still_missing} remaining unfillable rows.")

        elif strategy == "flag_for_review":
            report["actions_taken"].append({
                "column": date_col, "action": "flagged",
                "reason": f"{missing_pct}% missing dates -- too many to drop or fill confidently."})
            fb.append(f"[DATE] WARNING: {missing_pct}% dates missing in '{date_col}'. Flagged for manual review.")

    report["rows_after"] = len(df)
    fb.append(f"[DATE] Date integrity complete. {report['rows_before']} -> {len(df)} rows.")
    logger.info(f">>> [DateIntegrity] {report['rows_before']}->{len(df)} rows.")

    final_path = _save_parquet(df, config, "working_data.parquet")

    result_data = {
        # "cleaned_dataframe": df,
        "working_df_path": final_path,
        "date_integrity_report": report,
        "current_stage": "data_state_awareness",
        # "agent_feedback": fb,
        "steps_log": fb,
    }
    return _clean_for_serialization(result_data)


# --- Node 5: Data State Awareness ---

@observe(name="DataStateAwareness")
def data_state_awareness_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    # df_j = _df_json(state)
    current_path = state.get("working_df_path")
    existing = detect_existing_metrics.invoke({"df_path": current_path}) # "df_json": df_j → "df_path": current_path로 변경
    profiles = profile_all_columns.invoke({"df_path": current_path}) # "df_json": df_j → "df_path": current_path로 변경

    # 데이터 상태 판단 로직
    n = len(existing["existing_metrics"])
    data_state = "partially_processed" if n > 2 else ("mixed" if n > 0 else "raw")
    fb = _fb(state, f"[STAGE 1] Data state: {data_state}. {n} pre-existing metrics.")
    logger.info(f">>> [DataStateAwareness] data_state={data_state}, {n} existing metrics")
    
    result_data = {
        "data_state_report": {
            "data_state": data_state,
            "existing_metrics": existing["existing_metrics"],
            "profiles": profiles
        },
        "current_stage": "measurement_reconstruction",
        "steps_log": fb
    }
    return _clean_for_serialization(result_data)


# --- Node 6: Measurement Reconstruction ---

@observe(name="MeasurementReconstruction")
def measurement_reconstruction_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    profiles = state.get("data_state_report", {}).get("profiles", {})
    if not profiles:
        current_path = state.get("working_df_path")
        profiles = profile_all_columns.invoke({"df_path": current_path}) # "df_json": _df_json(state) → "df_path": current_path로 변경
    roles, justifications = {}, {}
    layers = {"volume": [], "performance": [], "efficiency": [], "value": [],
              "segment": [], "time": [], "other": []}
    for col, prof in profiles.items():
        r = classify_column_role.invoke({"col_name": col, "profile": prof})
        role = r["role"]; roles[col] = role; justifications[col] = r["justification"]
        if role.startswith("volume"): layers["volume"].append(col)
        elif role == "performance_rate": layers["performance"].append(col)
        elif role in ("efficiency_metric", "cost_metric"): layers["efficiency"].append(col)
        elif role == "value_metric": layers["value"].append(col)
        elif "dimension" in role: layers["segment" if "segment" in role else "time"].append(col)
        else: layers["other"].append(col)
    fb = _fb(state, f"[STAGE 2] Classified {len(roles)} columns.")
    logger.info(f">>> [MeasurementReconstruction] Classified {len(roles)} columns")
    result_data = {"column_roles": roles, "role_justifications": justifications,
            "detected_layers": layers, "current_stage": "metric_derivation", "steps_log": fb}
    return _clean_for_serialization(result_data)

# --- Node 7: Metric Derivation ---

@observe(name="MetricDerivation")
def metric_derivation_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    # df = _df(state); df_j = _df_json(state)
    current_path = state.get("working_df_path")
    
    roles = state.get("column_roles", {})
    existing = state.get("data_state_report", {}).get("existing_metrics", {})
    vol_exp = [c for c, r in roles.items() if r == "volume_exposure" and c in state.get("column_roles").keys()] # df.columns → state.get("column_roles").keys() 교체
    vol_eng = [c for c, r in roles.items() if r == "volume_engagement" and c in state.get("column_roles").keys()] # df.columns → state.get("column_roles").keys() 교체
    vol_out = [c for c, r in roles.items() if r == "volume_outcome" and c in state.get("column_roles").keys()] # df.columns → state.get("column_roles").keys() 교체
    cost_cols = [c for c, r in roles.items() if r == "cost_metric" and c in state.get("column_roles").keys()] # df.columns → state.get("column_roles").keys() 교체
    rev_cols = [c for c, r in roles.items() if r == "value_metric" and c in state.get("column_roles").keys()] # df.columns → state.get("column_roles").keys() 교체

    recipes = []
    for imp in vol_exp:
        for clk in vol_eng:
            recipes.append(("CTR", clk, imp, "rate", "Is the ad attracting attention?", "performance"))
    for clk in vol_eng:
        for conv in vol_out:
            recipes.append(("CVR", conv, clk, "rate", "Is the offer convincing?", "performance"))
    for imp in vol_exp:
        for conv in vol_out:
            recipes.append(("overall_conv_rate", conv, imp, "rate", "What fraction converts?", "performance"))
    for cc in cost_cols:
        for clk in vol_eng:
            recipes.append(("CPC", cc, clk, "eff", "Cost per click?", "efficiency"))
    for cc in cost_cols:
        for conv in vol_out:
            recipes.append(("CPA", cc, conv, "eff", "Cost per result?", "efficiency"))
    for rc in rev_cols:
        for cc in cost_cols:
            recipes.append(("ROAS", rc, cc, "eff", "Is spending profitable?", "efficiency"))
    for rc in rev_cols:
        for conv in vol_out:
            recipes.append(("AOV", rc, conv, "eff", "Value per conversion?", "value"))

    derived, skipped = [], []
    for name, num, den, kind, biz_q, layer in recipes:
        if name in existing:
            skipped.append({"metric": name, "reason": f"Already exists as '{existing[name]}'."}); continue
        if kind == "rate":
            r = derive_rate_metric.invoke({"df_path": current_path, "numerator_col": num, # "df_json": df_j → "df_path": current_path 교체
                                           "denominator_col": den, "metric_name": name,
                                           "business_question": biz_q})
        else:
            r = derive_efficiency_metric.invoke({"df_path": current_path, "cost_col": num, # "df_json": df_j → "df_path": current_path 교체
                                                  "event_col": den, "metric_name": name,
                                                  "business_question": biz_q})
        r["layer"] = layer; derived.append(r)

    fb = _fb(state, f"[STAGE 3-4] Derived {len(derived)} metrics, skipped {len(skipped)}.")
    logger.info(f">>> [MetricDerivation] Derived {len(derived)} metrics, skipped {len(skipped)}")
    result_data = {"derived_metrics": derived, "derivation_skipped": skipped,
            "current_stage": "reliability_signals", "steps_log": fb}
    return _clean_for_serialization(result_data)

# --- Node 8: Reliability Signals ---

@observe(name="ReliabilitySignals")
def reliability_signals_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    # df_j = _df_json(state)
    current_path = state.get("working_df_path")

    derived = state.get("derived_metrics", [])
    flags = {}
    for d in derived:
        formula = d.get("formula", "")
        if "/" in formula:
            denom = formula.split("/")[-1].strip()
            r = add_reliability_flags.invoke({"df_path": current_path, "denominator_col": denom}) # "df_json": df_j → "df_path": current_path 교체
            if "error" not in r: flags[d["metric_name"]] = r
    fb = _fb(state, f"[STAGE 5] Reliability flags for {len(flags)} metrics.")
    logger.info(f">>> [ReliabilitySignals] Flags for {len(flags)} metrics")
    result_data = {"reliability_flags": flags, "current_stage": "semantic_cleanup", "steps_log": fb}
    return _clean_for_serialization(result_data)


# --- Node 9: Semantic Cleanup ---

@observe(name="SemanticCleanup")
def semantic_cleanup_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    roles = state.get("column_roles", {}); actions = []
    for col, role in roles.items():
        if role == "noise_artifact":
            actions.append({"action": "drop", "column": col, "reason": ">70% zeros."})
        elif role == "identifier":
            actions.append({"action": "keep_as_index", "column": col, "reason": "Identifier."})
    fb = _fb(state, f"[STAGE 6] Cleanup: {len(actions)} actions.")
    logger.info(f">>> [SemanticCleanup] {len(actions)} actions")
    result_data = {"cleanup_actions": actions, "current_stage": "funnel_leakage", "steps_log": fb}
    return _clean_for_serialization(result_data)


# --- Node 10: Funnel & Leakage ---

@observe(name="FunnelLeakage")
def funnel_leakage_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    # df = _df(state)
    current_path = state.get("working_df_path")
    
    roles = state.get("column_roles", {})
    vol_exp = [c for c, r in roles.items() if r == "volume_exposure" and c in state.get("column_roles").keys()] # df.columns → state.get("column_roles").keys() 교체
    vol_eng = [c for c, r in roles.items() if r == "volume_engagement" and c in state.get("column_roles").keys()] # df.columns → state.get("column_roles").keys() 교체
    vol_out = [c for c, r in roles.items() if r == "volume_outcome" and c in state.get("column_roles").keys()] # df.columns → state.get("column_roles").keys() 교체
    funnel_cols = vol_exp + vol_eng + vol_out
    leakage = {}
    if len(funnel_cols) >= 2:
        leakage = compute_funnel_leakage.invoke({"df_path": current_path, "funnel_cols": funnel_cols}) # "df_json": df_j → "df_path": current_path 교체
    fb = _fb(state, f"[STAGE 11] Funnel: {funnel_cols}. Worst: {leakage.get('_worst_stage', 'N/A')}.")
    logger.info(f">>> [FunnelLeakage] Funnel cols: {funnel_cols}")
    result_data = {"funnel_chains": [funnel_cols] if funnel_cols else [], "leakage_analysis": leakage,
            "current_stage": "context_enrichment", "steps_log": fb}
    return _clean_for_serialization(result_data)


# --- Node 11: Context Enrichment ---

@observe(name="ContextEnrichment")
def context_enrichment_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    # df = _df(state); df_j = _df_json(state); 
    df = _load_parquet(state)
    current_path = state.get("working_df_path")

    roles = state.get("column_roles", {})
    layers = state.get("detected_layers", {}); fb = []

    cost_cols = [c for c, r in roles.items() if r == "cost_metric" and c in df.columns]
    out_cols = [c for c, r in roles.items() if r == "volume_outcome" and c in df.columns]
    imp_cols = [c for c, r in roles.items() if r == "volume_exposure" and c in df.columns]
    rev_cols = [c for c, r in roles.items() if r == "value_metric" and c in df.columns]

    share_metrics = []
    for role_key in ["cost_metric", "value_metric", "volume_outcome"]:
        for c in [c for c, r in roles.items() if r == role_key and c in df.columns]:
            sm = compute_share_metrics.invoke({"df_path": current_path, "value_col": c}) # "df_json": df_j → "df_path": current_path 교체
            if "error" not in sm: share_metrics.append(sm)

    vol_eff_flags = []
    for cc in cost_cols:
        for oc in out_cols:
            hc = df[cc] > df[cc].quantile(0.75); lo = df[oc] < df[oc].quantile(0.25)
            n = int((hc & lo).sum())
            if n > 0: vol_eff_flags.append({"flag": "high_spend_low_outcome", "cost_col": cc, "outcome_col": oc, "affected_rows": n})

    time_cols = layers.get("time", []); has_time = len(time_cols) > 0; trend_metrics = []
    if has_time:
        tc = time_cols[0]
        for mc in cost_cols + out_cols + rev_cols:
            t = compute_period_change.invoke({"df_path": current_path, "value_col": mc, "time_col": tc}) # "df_json": df_j → "df_path": current_path 교체
            if "error" not in t: trend_metrics.append(t)

    saturation_flags = []
    for cc in cost_cols:
        for conv in out_cols:
            for imp in imp_cols:
                sat = detect_saturation.invoke({"df_path": current_path, "cost_col": cc, "conversion_col": conv, "impression_col": imp}) # "df_json": df_j → "df_path": current_path 교체
                if sat.get("diminishing_return_detected"): saturation_flags.append(sat)

    obj = {"inferred": "performance-oriented" if out_cols and cost_cols else ("awareness-oriented" if imp_cols else "unknown"),
           "note": "Inferred from available columns."}

    safety = [{"metric": d["metric_name"], "numerator": d["formula"].split("/")[0].strip(),
               "denominator": d["formula"].split("/")[1].strip()}
              for d in state.get("derived_metrics", []) if "/" in d.get("formula", "")]

    anomalies = detect_anomalies.invoke({"df_path": current_path, "roles": roles}) # "df_json": df_j → "df_path": current_path 교체

    fb.append(f"[CONTEXT] Shares: {len(share_metrics)}, Flags: {len(vol_eff_flags)}, "
              f"Trends: {len(trend_metrics)}, Saturation: {len(saturation_flags)}, Anomalies: {len(anomalies)}.")
    logger.info(f">>> [ContextEnrichment] Shares={len(share_metrics)}, Trends={len(trend_metrics)}, "
                f"Saturation={len(saturation_flags)}, Anomalies={len(anomalies)}")

    result_data = {
        "share_metrics": share_metrics, "volume_efficiency_flags": vol_eff_flags,
        "has_time_dimension": has_time, "trend_metrics": trend_metrics,
        "saturation_flags": saturation_flags, "objective_notes": obj,
        "safety_checks": safety, "anomaly_flags": anomalies,
        "current_stage": "final_assembly", "steps_log": fb
    }
    return _clean_for_serialization(result_data)


# --- Node 12: Final Assembly ---

@observe(name="FinalAssembly")
def final_assembly_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    # --- 기존 로직 --- 
    # df = _df(state).copy()
    df = _load_parquet(state)

    roles = state.get("column_roles", {})
    derived = state.get("derived_metrics", []); reliability = state.get("reliability_flags", {})
    anomalies = state.get("anomaly_flags", []); saturation = state.get("saturation_flags", [])
    vol_eff_flags = state.get("volume_efficiency_flags", []); share_metrics = state.get("share_metrics", [])
    catalogue = []

    # Derive columns
    for d in derived:
        formula = d.get("formula", ""); name = d["metric_name"]
        if "/" in formula:
            parts = [p.strip() for p in formula.split("/")]
            if parts[0] in df.columns and parts[1] in df.columns:
                df[name] = df[parts[0]] / df[parts[1]].replace(0, np.nan)
                catalogue.append({"feature_name": name, "formula": formula,
                                  "layer": d.get("layer", "derived"),
                                  "business_question": d.get("business_question", ""),
                                  "stabilization": "none (real business value)"})

    # Share columns
    for sm in share_metrics:
        col_name = sm.get("metric", ""); formula = sm.get("formula", "")
        if formula and "/" in formula:
            src = formula.split("/")[0].strip()
            if src in df.columns:
                total = df[src].sum()
                if total > 0:
                    df[col_name] = df[src] / total
                    catalogue.append({"feature_name": col_name, "formula": formula, "layer": "contribution",
                                      "business_question": "Contribution share?", "stabilization": "none"})

    # Reliability flags
    for metric, info in reliability.items():
        denom = info.get("denominator", ""); thresh = info.get("threshold_used", 30)
        if denom in df.columns:
            f1 = f"{metric}_low_volume_flag"; df[f1] = (df[denom] < thresh).astype(int)
            f2 = f"{metric}_zero_div_flag"; df[f2] = (df[denom] == 0).astype(int)
            catalogue.append({"feature_name": f1, "formula": f"{denom} < {thresh}",
                              "layer": "reliability", "business_question": "Trust this ratio?",
                              "stabilization": "binary flag"})
            catalogue.append({"feature_name": f2, "formula": f"{denom} == 0",
                              "layer": "reliability", "business_question": "Metric computable?",
                              "stabilization": "binary flag"})

    # Anomaly flag
    if anomalies:
        df["anomaly_flag"] = 0
        for a in anomalies:
            cols = a.get("columns", [])
            if len(cols) == 2 and all(c in df.columns for c in cols):
                mask = (df[cols[0]] > df[cols[0]].quantile(0.75)) & (df[cols[1]] == 0)
                df.loc[mask, "anomaly_flag"] = 1
        catalogue.append({"feature_name": "anomaly_flag", "formula": "composite",
                          "layer": "reliability", "business_question": "Investigate this row?",
                          "stabilization": "binary flag"})

    # Saturation flag
    if saturation:
        df["diminishing_return_flag"] = 0
        for sat in saturation:
            cc = sat.get("cost_col", "")
            if cc in df.columns: df.loc[df[cc] > df[cc].quantile(0.75), "diminishing_return_flag"] = 1
        catalogue.append({"feature_name": "diminishing_return_flag", "formula": "cost up + conv rate down",
                          "layer": "reliability", "business_question": "Diminishing returns?",
                          "stabilization": "binary flag"})

    # Volume/efficiency flags
    for vef in vol_eff_flags:
        cc, oc = vef["cost_col"], vef["outcome_col"]
        fn = f"high_spend_low_outcome_{cc}_{oc}"
        if cc in df.columns and oc in df.columns:
            df[fn] = ((df[cc] > df[cc].quantile(0.75)) & (df[oc] < df[oc].quantile(0.25))).astype(int)

    # Order columns
    seg = [c for c, r in roles.items() if "dimension" in r and c in df.columns]
    vol = [c for c, r in roles.items() if r.startswith("volume") and c in df.columns]
    perf = [c for c in df.columns if c in [d["metric_name"] for d in derived if d.get("layer") == "performance"]]
    cc_list = [c for c, r in roles.items() if r == "cost_metric" and c in df.columns]
    vc_list = [c for c, r in roles.items() if r == "value_metric" and c in df.columns]
    eff = [c for c in df.columns if c in [d["metric_name"] for d in derived if d.get("layer") == "efficiency"]]
    share_c = [c for c in df.columns if "_share" in c]
    flag_c = [c for c in df.columns if "flag" in c.lower()]
    ordered = seg + vol + perf + cc_list + vc_list + eff + share_c + flag_c
    for c in df.columns:
        if c not in ordered: ordered.append(c)
    
    # df = df[[c for c in ordered if c in df.columns]]

    # fb = _fb(state, f"[ASSEMBLY] Final: {df.shape}. {len(catalogue)} features catalogued.")
    # logger.info(f">>> [FinalAssembly] Final shape: {df.shape}, {len(catalogue)} features")
    # return {"reporting_dataframe": df, "feature_catalogue": catalogue, "column_order": list(df.columns),
    #         "current_stage": "output_formatting",
    #         "agent_feedback": fb, "steps_log": fb}

    final_path = _save_parquet(df, config, "clean_data.parquet")
    
    #  로깅에 남기기
    fb = _fb(state, f"[ASSEMBLY] Final: {df.shape}. Parquet saved to {final_path}.")
    logger.info(f">>> [FinalAssembly] Final shape: {df.shape}, saved to: {final_path}")

    # Return 값 수정: DataFrame 본체가 아닌 '경로 문자열' 전달 (working_df_path)
    result_data = {
        "working_df_path": final_path, # <--- 여기가 핵심! df 대신 save_path를 전달
        "feature_catalogue": catalogue, 
        "column_order": list(df.columns),
        "current_stage": "output_formatting",
        "steps_log": fb
    }
    return _clean_for_serialization(result_data)


# --- Node 13: Output Formatting ---

@observe(name="OutputFormatting")
def output_formatting_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    # --- 기존 로직 ---
    #  rdf = state.get("reporting_dataframe")
    #  if not isinstance(rdf, pd.DataFrame): rdf = pd.read_json(io.StringIO(rdf), orient="split")

    rdf_path = state.get("working_df_path")
    original_shape = [0,0]
    original_columns = []

    try:
        if isinstance(rdf_path, str) and os.path.exists(rdf_path):
            # 파일이 정상적으로 존재하면 파케이 파일을 읽어들입니다.
            # 10줄만 읽어서 포맷팅에 쓸 거라면 head(10)을 추천합니다. (미리보기용 객체)
            sample_df = pd.read_parquet(rdf_path).head(10) 
            
            # 2. 파일 메타데이터만 읽어오기 (전체 데이터 안 올림!)
            parquet_metadata = pq.read_metadata(rdf_path)
            original_shape = [parquet_metadata.num_rows, parquet_metadata.num_columns]
            original_columns = parquet_metadata.schema.names
        elif isinstance(rdf_path, pd.DataFrame): 
             # (혹시 몰라서 남겨두는 예외 처리) 만약 원래 객체가 들어왔다면
             sample_df = rdf_path.head(10)
             original_shape = list(rdf_path.shape)         # 원본 전체 모양
             original_columns = list(rdf_path.columns)     # 원본 전체 열
        else:
            # 경로도 아니고 DataFrame도 아니면 빈 프레임 할당
            sample_df = pd.DataFrame()
            original_shape = [0, 0]
            original_columns = []
            
    except Exception as e:
        # 에러가 나면 여기서 멈추지 말고 빈 프레임으로 대체해서 넘어감
        logger.error(f"DataFrame 로드 중 에러 발생: {e}")
        sample_df = pd.DataFrame()
        shape_info = "shape: error loading preview"

    fmt = state.get("output_format", "agent_prompt")
    catalogue = state.get("feature_catalogue", [])
    leakage = state.get("leakage_analysis", {})
    objective = state.get("objective_notes", {})
    anomalies = state.get("anomaly_flags", [])
    reliability = state.get("reliability_flags", {})
    trends = state.get("trend_metrics", [])
    saturation = state.get("saturation_flags", [])
    cat_report = state.get("categorical_standardization_report", {})
    dedup_report = state.get("duplicate_cleanup_report", {})
    date_report = state.get("date_integrity_report", {})
    raw_prep_report = state.get("raw_preprocessing_report", {})

    metadata = {
        "shape": original_shape, "columns": original_columns,
        "column_roles": state.get("column_roles", {}),
        "feature_catalogue": catalogue,
        "data_integrity": {"raw_preprocessing": raw_prep_report,
                           "categorical_standardization": cat_report,
                           "duplicate_cleanup": dedup_report,
                           "date_integrity": date_report},
        "funnel_leakage": leakage, "objective_inference": objective,
        "anomalies": anomalies, "reliability_flags": reliability,
        "trend_metrics": trends, "saturation_signals": saturation,
    }

    # if fmt == "json":
    #     output = json.dumps({"metadata": metadata,
    #                          "data": json.loads(rdf.to_json(orient="records"))}, indent=2, default=str)
    # elif fmt == "csv":
    #     output = f"# Marketing Report\n# Shape: {rdf.shape}\n" + rdf.to_csv(index=False)
    # elif fmt == "markdown":
    #     lines = ["# Marketing Measurement Report", "",
    #              f"**Dataset**: {rdf.shape[0]} rows x {rdf.shape[1]} columns",
    #              f"**Objective**: {objective.get('inferred', 'unknown')}", "",
    #              "## Data Integrity Actions", ""]
    #     for a in date_report.get("actions_taken", []):
    #         lines.append(f"- Date: {a.get('action', '?')} on {a.get('column', '?')}")
    #     lines += ["", "## Feature Catalogue", "",
    #               "| Feature | Formula | Layer | Business Question |",
    #               "|---------|---------|-------|-------------------|"]
    #     for f_item in catalogue:
    #         lines.append(f"| {f_item['feature_name']} | {f_item['formula']} | {f_item['layer']} | {f_item['business_question']} |")
    #     lines += ["", "## Data Preview", "", rdf.head(10).to_markdown(index=False)]
    #     output = "\n".join(lines)
    # else:  # agent_prompt
    #     sections = [
    #         "<MARKETING_REPORT_DATA>",
    #         "<INSTRUCTIONS>",
    #         "You are receiving structured marketing measurement data from an upstream agent.",
    #         "Generate a human-readable marketing performance report.",
    #         "Data is structured: [segments] > [volume] > [rates] > [cost] > [value] > [efficiency] > [flags]",
    #         "RULES:",
    #         "- Check reliability flags before drawing conclusions.",
    #         "- low_volume_flag=1 means unstable ratio.",
    #         "- anomaly_flag=1 means investigate.",
    #         "- Use real values only. Do NOT smooth or transform.",
    #         "</INSTRUCTIONS>",
    #         "",
    #         "<DATA_INTEGRITY_REPORT>",
    #         json.dumps(metadata["data_integrity"], indent=2, default=str),
    #         "</DATA_INTEGRITY_REPORT>",
    #         "",
    #         "<METADATA>",
    #         f"shape: {rdf.shape}",
    #         f"objective: {objective.get('inferred', 'unknown')}",
    #         f"anomalies: {len(anomalies)}",
    #         f"saturation_signals: {len(saturation)}",
    #         "</METADATA>",
    #         "",
    #         "<FEATURE_CATALOGUE>",
    #         json.dumps(catalogue, indent=2, default=str),
    #         "</FEATURE_CATALOGUE>",
    #         "",
    #         "<FUNNEL_LEAKAGE>",
    #         json.dumps(leakage, indent=2, default=str),
    #         "</FUNNEL_LEAKAGE>",
    #         "",
    #         "<ANOMALIES>",
    #         json.dumps(anomalies, indent=2, default=str),
    #         "</ANOMALIES>",
    #         "",
    #         "<RELIABILITY_FLAGS>",
    #         json.dumps(reliability, indent=2, default=str),
    #         "</RELIABILITY_FLAGS>",
    #         "",
    #         "<TREND_METRICS>",
    #         json.dumps(trends, indent=2, default=str),
    #         "</TREND_METRICS>",
    #         "",
    #         '<DATA format="json_records">',
    #         rdf.to_json(orient="records", indent=2),
    #         "</DATA>",
    #         "</MARKETING_REPORT_DATA>",
    #     ]
    #     output = "\n".join(sections)

    # --- 포멧팅 ---
    if fmt == "agent_prompt":
        sections = [
            "<MARKETING_REPORT_DATA>",
            "<INSTRUCTIONS>",
            # ... 기존 안내문 동일 ...
            "</INSTRUCTIONS>",
            # ... (중간의 메타데이터 블록 동일) ...
            
            # --- 핵심 변경점 ---
            # 여기서 전체 <DATA_RECORDS>를 넣지 않습니다. 대신 저장된 경로를 알려줍니다.
            "<DATA_REFERENCE>",
            f"The fully cleaned and preprocessed data is securely saved as a Parquet file at:",
            f"{rdf_path}",
            "You can use pandas to read this file: pd.read_parquet('path')",
            "</DATA_REFERENCE>",
            
            "<DATA_PREVIEW>",
            sample_df.to_csv(index=False),
            "</DATA_PREVIEW>",
            "</MARKETING_REPORT_DATA>"
        ]
        output = "\n".join(sections)
    
    # JSON 포맷팅일 경우에도 전체 데이터를 빼버립니다.
    elif fmt == "json":
        output = json.dumps({
            "metadata": metadata, # metadata 변수 구성은 기존 코드 참고
            "data_path": rdf_path, # 전체 데이터 배열 대신 주소만 넘김
            "preview": json.loads(sample_df.to_json(orient="records"))
        }, indent=2, default=str)   
    

    fb = _fb(state, f"[OUTPUT] Formatted as '{fmt}' ({len(output):,} chars).")
    logger.info(f">>> [OutputFormatting] Formatted as '{fmt}' ({len(output):,} chars)")
    # return {"formatted_output": output, "current_stage": "quality_gate",
    #         "agent_feedback": fb, "steps_log": fb}
    result_data = {
        "formatted_output": output,
        "current_stage": "quality_gate",
        "steps_log": fb
    }
    return _clean_for_serialization(result_data)

# --- Node 14: Quality Gate ---

@observe(name="QualityGate")
def quality_gate_node(state: preprocessState, config: RunnableConfig) -> preprocessState:
    catalogue = state.get("feature_catalogue", [])
    derived = state.get("derived_metrics", [])
    reliability = state.get("reliability_flags", {})
    iteration = state.get("iteration_count", 0) + 1
    issues = []

    # 범용성을 위해 규칙 완화 (주석처리): 파생변수, 신뢰성 플래그 없어도 통과
    # if not derived: issues.append("No metrics derived.")
    # if not reliability: issues.append("No reliability flags.")

    rdf = state.get("working_df_path")

    # if isinstance(rdf, pd.DataFrame):
    #     cols = set(rdf.columns)
    #     for d in derived:
    #         if "/" in d.get("formula", ""):
    #             for p in d["formula"].split("/"):
    #                 if p.strip() not in cols:
    #                     issues.append(f"Base column '{p.strip()}' missing from output.")

    # 1. 검사용 컬럼 셋 초기화
    cols = set() 
    # 2. 케이스별로 컬럼 이름들(names) 추출
    if isinstance(rdf, str) and os.path.exists(rdf):
        # 파케이 파일인 경우: 메타데이터만 읽어서 컬럼명 추출
        cols = set(pq.read_metadata(rdf).schema.names)
    elif isinstance(rdf, pd.DataFrame):
        # 혹시 모를 기존 데이터프레임용 (하위 호환성)
        cols = set(rdf.columns)
    # 3. 만약 컬럼을 하나도 못 찾았다면 검증 실패로 간주
    if not cols and (derived or reliability):
         issues.append("Could not load column names for validation.")
    # 4. cols를 사용해서 검증 수행
    for d in derived:
        if "/" in d.get("formula", ""):
            for p in d["formula"].split("/"):
                if p.strip() not in cols:
                    issues.append(f"Base column '{p.strip()}' missing from output.")

    # LLM Review
    llm_review = ""
    try:
        configurable = config.get("configurable", {})
        u_id = configurable.get("user_id")
        s_id = configurable.get("session_id")
        node_conf = state.get("node_models", {}).get("eval_node", {})  
        provider = node_conf.get("provider") or "google"   
        model_name = node_conf.get("model") or "gemma-3-27b-it"
        llm, callbacks = LLMFactory.create(provider, model_name, temperature=0.3)
        prompt = (
            f"Review marketing dataset: {len(catalogue)} features, "
            f"{len(derived)} metrics, {len(reliability)} reliability flags. Issues?"
        )
        with langfuse_session(session_id=s_id, user_id=u_id) as lf_metadata:
            invoke_cfg = merge_runnable_config(
                config,
                callbacks=callbacks,
                metadata=lf_metadata,
            )
            response = llm.invoke(prompt, config=invoke_cfg)
        llm_review = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        llm_review = f"LLM review failed: {e}"

    fb = _fb(state, f"[QUALITY GATE iter {iteration}] Issues: {len(issues)}.")
    for i in issues: fb.append(f"   X {i}")
    if llm_review: fb.append(f"   LLM: {llm_review[:400]}")
    logger.info(f">>> [QualityGate] iter={iteration}, issues={len(issues)}")

    if issues and iteration < 3:
        # return {"current_stage": "data_state_awareness", "iteration_count": iteration,
                # "agent_feedback": fb, "steps_log": fb}
        result_data = {
            "current_stage": "data_state_awareness",
            "iteration_count": iteration,
            "steps_log": fb
        }
        return _clean_for_serialization(result_data)

    fb.append("PIPELINE COMPLETE.")
    logger.info(">>> [QualityGate] PIPELINE COMPLETE")

    # parquet를 사용하는 방향으로 수정해 아래 내용은 더 이상 사용하지 않습니다.
    # Finalize: save clean CSV + build clean_data dict
    # final_df = None
    # if isinstance(rdf, pd.DataFrame):
    #     final_df = rdf
    # else:
    #     try:
    #         final_df = _df(state)
    #     except ValueError:
    #         pass
    # clean_file_path = state.get("clean_file_path", "")
    # clean_data = None
    # if final_df is not None:
    #     # 최종 데이터로 clean CSV 업데이트
    #     clean_file_path = _save_clean_csv(state, final_df)
    #     # AgentState 의 clean_data 는 dict 형태
    #     try:
    #         clean_data = json.loads(final_df.to_json(orient="split"))
    #     except Exception:
    #         clean_data = {"columns": list(final_df.columns), "shape": list(final_df.shape)}
    # return {
    #     "working_df_path": rdf,
    #     "current_stage": "done",
    #     "iteration_count": iteration,
    #     "agent_feedback": fb,
    #     "steps_log": fb,                  # Main_node 에서 steps_log 를 읽음
    #     "clean_file_path": clean_file_path,
    #     "clean_data": clean_data,          # Main_node 에서 clean_data 를 읽음
    # }
    
    result_data = {
        "current_stage": "done",
        "iteration_count": iteration,
        "steps_log": fb
    }
    return _clean_for_serialization(result_data)
    
# parquet를 사용하는 방향으로 수정해 아래 내용은 더 이상 사용하지 않습니다.
# --- Helper: Save Clean CSV ---
# def _save_clean_csv(state: preprocessState, df: pd.DataFrame) -> str:
#     """정제 완료된 DataFrame을 {원본명}_clean.csv 로 저장."""
#     original_path = state.get("file_path", "")
#     if not original_path:
#         return ""
#     base, ext = os.path.splitext(original_path)
#     clean_file_path = f"{base}_clean.csv"
#     try:
#         df.to_csv(clean_file_path, index=False)
#         logger.info(f">>> 정제 CSV 저장: {clean_file_path}")
#     except Exception as e:
#         logger.error(f"CSV 저장 실패: {e}")
#         clean_file_path = original_path
#     return clean_file_path

# --- Router ---
def route_after_quality_gate(state: preprocessState) -> str:
    """Quality gate 분기: 이슈가 있으면 data_state_awareness로 루프, 없으면 END."""
    if state.get("current_stage") == "data_state_awareness":
        return "data_state_awareness"
    return END