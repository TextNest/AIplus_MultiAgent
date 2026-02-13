import pandas as pd


def get_df_summary(df: pd.DataFrame):

    summary = []
    summary.append(f"- 데이터 형태 (Shape): {df.shape[0]}행, {df.shape[1]}열")

    # 1. 컬럼명과 데이터 타입
    summary.append("\n- 컬럼 정보 및 타입:")
    summary.append(df.dtypes.to_string())

    # 2. 결측치 정보
    null_info = df.isnull().sum()
    if null_info.sum() > 0:
        summary.append("\n- 결측치 수:")
        summary.append(null_info[null_info > 0].to_string())
    else:
        summary.append("\n- 결측치 없음")

    # 3. 범주형 데이터의 고유값(Unique) 개수
    # 고유값이 너무 많지 않은(예: 20개 이하) 컬럼들만 골라 정보를 줍니다.
    summary.append("\n- 주요 범주형 데이터 정보:")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count <= 20:
            summary.append(
                f"  * {col}: {unique_count}개의 고유값 ({df[col].unique()[:5]}...)"
            )
        else:
            summary.append(f"  * {col}: {unique_count}개의 고유값")

    # 4. 수치형 데이터 통계 요약
    summary.append("\n- 수치 데이터 요약 (describe):")
    summary.append(df.describe().loc[["mean", "min", "max"]].to_string())

    # 5. 실제 데이터 샘플 (Top 3)
    summary.append("\n- 데이터 샘플 (Top 3):")
    summary.append(df.head(3).to_string())

    return "\n".join(summary)