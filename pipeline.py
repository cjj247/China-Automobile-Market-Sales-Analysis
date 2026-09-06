"""中国汽车市场销量分析：真实公开数据的清洗、指标计算与留痕输出。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "china_auto_matplotlib")
)

import matplotlib.pyplot as plt

RAW_FILE = PROJECT_ROOT / "data" / "China Automobile Sales Data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data"
TABLEAU_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
LOG_DIR = PROJECT_ROOT / "run_logs"

SOURCE_PAGE = (
    "https://www.kaggle.com/datasets/felixzhao/"
    "china-automobile-monthly-sales-data-2018-2024-4"
)
SOURCE_RAW_MIRROR = (
    "https://github.com/hoangphuc-ngo/Forecasting-China-EV-Sales/raw/"
    "refs/heads/main/Data/Raw/China%20Automobile%20Sales%20Data.csv"
)

BODY_TYPE_CN = {
    "SUV": "SUV",
    "Sedan": "轿车",
    "Hatchback": "两厢车",
    "MPV": "MPV",
    "Sports Car": "跑车",
    "Unknown": "未知",
}

COUNTRY_CN = {
    "China": "中国",
    "Japan": "日本",
    "Germany": "德国",
    "United States": "美国",
    "South Korea": "韩国",
    "France": "法国",
    "Czech Republic": "捷克",
    "Sweden": "瑞典",
    "United Kingdom": "英国",
    "Italy": "意大利",
}

BRAND_NORMALIZATION = {
    "Shenlan": "Deepal",
    "Byton": "Denza",
    "LanTu": "Voyah",
}

EV_ONLY_BRANDS = {
    "Aiways",
    "Deepal",
    "Denza",
    "GAC Aion",
    "Geometry",
    "Hozon",
    "Leapmotor",
    "Li Auto",
    "NIO",
    "Ora",
    "Polestar",
    "SERES",
    "Tesla",
    "Voyah",
    "WM Motor",
    "XPeng Motors",
    "Zeekr",
}


def ensure_directories() -> None:
    for directory in (PROCESSED_DIR, TABLEAU_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("china_auto_pipeline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_text(value: object) -> str:
    """使用 re 去除首尾空格并合并连续空白。"""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_raw(logger: logging.Logger) -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"未找到原始数据：{RAW_FILE}")
    frame = pd.read_csv(RAW_FILE, encoding="utf-8")
    logger.info("读取原始数据：%s 行，%s 列", len(frame), len(frame.columns))
    logger.info("原始文件 SHA256：%s", sha256(RAW_FILE))
    return frame


def build_quality_report(raw: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(raw["year_month"], errors="coerce")
    numeric_sales = pd.to_numeric(raw["units_sold"], errors="coerce")
    low_price = pd.to_numeric(raw["low_price"], errors="coerce")
    high_price = pd.to_numeric(raw["high_price"], errors="coerce")
    checks = [
        ("原始行数", len(raw), "记录"),
        ("原始字段数", len(raw.columns), "字段"),
        ("完全重复行", int(raw.duplicated().sum()), "记录"),
        (
            "车型-厂商-月份重复组合",
            int(raw.duplicated(["model", "make", "year_month"]).sum()),
            "记录",
        ),
        ("车型类别缺失", int(raw["body_type"].isna().sum()), "记录"),
        ("日期无法解析", int(dates.isna().sum()), "记录"),
        ("销量无法转为数值", int(numeric_sales.isna().sum()), "记录"),
        ("销量小于等于0", int((numeric_sales <= 0).sum()), "记录"),
        ("价格无法转为数值", int((low_price.isna() | high_price.isna()).sum()), "记录"),
        ("最低价高于最高价", int((low_price > high_price).sum()), "记录"),
        ("价格为0", int(((low_price == 0) | (high_price == 0)).sum()), "记录"),
    ]
    return pd.DataFrame(checks, columns=["检查项目", "数量", "单位"])


def clean_and_engineer(raw: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    expected_columns = [
        "model",
        "units_sold",
        "make",
        "low_price",
        "high_price",
        "year_month",
        "is_ev",
        "body_type",
        "brand",
        "brand_country",
    ]
    if raw.columns.tolist() != expected_columns:
        raise ValueError(f"原始字段发生变化：{raw.columns.tolist()}")

    frame = raw.copy()
    before = len(frame)
    frame = frame.drop_duplicates().copy()
    logger.info("删除完全重复行：%s 条", before - len(frame))

    text_columns = ["model", "make", "is_ev", "body_type", "brand", "brand_country"]
    for column in text_columns:
        frame[column] = frame[column].fillna("Unknown").map(normalize_text)

    frame["brand_raw"] = frame["brand"]
    frame["is_ev_raw"] = frame["is_ev"]
    frame["brand"] = frame["brand"].replace(BRAND_NORMALIZATION)
    deepal_model = frame["brand"].eq("Deepal") & frame["model"].str.startswith("长安深蓝")
    frame.loc[deepal_model, "model"] = frame.loc[deepal_model, "model"].str.replace(
        r"^长安", "", regex=True
    )
    energy_correction = frame["brand"].isin(EV_ONLY_BRANDS) & ~frame["is_ev"].eq("EV")
    frame["energy_label_corrected"] = energy_correction
    frame.loc[energy_correction, "is_ev"] = "EV"
    logger.info("依据纯新能源品牌清单修正能源标签：%s 条", int(energy_correction.sum()))

    frame["year_month"] = pd.to_datetime(frame["year_month"], errors="coerce")
    for column in ("units_sold", "low_price", "high_price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_required = (
        frame["year_month"].isna()
        | frame["units_sold"].isna()
        | (frame["units_sold"] <= 0)
        | frame["model"].eq("")
        | frame["make"].eq("")
    )
    invalid_count = int(invalid_required.sum())
    frame = frame.loc[~invalid_required].copy()
    logger.info("删除关键字段无效记录：%s 条", invalid_count)

    frame["units_sold"] = frame["units_sold"].astype("int64")
    frame["low_price"] = frame["low_price"].round(1)
    frame["high_price"] = frame["high_price"].round(1)
    valid_price = (frame["low_price"] > 0) & (frame["high_price"] > 0)
    frame["avg_price_k_rmb"] = np.where(
        valid_price,
        (frame["low_price"] + frame["high_price"]) / 2,
        np.nan,
    )
    frame["avg_price_k_rmb"] = frame["avg_price_k_rmb"].round(1)

    frame["energy_type"] = frame["is_ev"].map(
        {"EV": "新能源", "Gasoline": "燃油"}
    ).fillna("未知")
    frame["body_type_cn"] = frame["body_type"].map(BODY_TYPE_CN).fillna(frame["body_type"])
    frame["country_cn"] = frame["brand_country"].map(COUNTRY_CN).fillna(
        frame["brand_country"]
    )
    frame["brand_origin"] = np.where(frame["brand_country"].eq("China"), "中国品牌", "海外品牌")
    frame["price_band"] = pd.cut(
        frame["avg_price_k_rmb"],
        bins=[-np.inf, 100, 150, 200, 300, np.inf],
        labels=["10万元以下", "10–15万元", "15–20万元", "20–30万元", "30万元以上"],
        ordered=True,
    ).astype("string").fillna("价格缺失")
    frame["year"] = frame["year_month"].dt.year.astype("int64")
    frame["month"] = frame["year_month"].dt.month.astype("int64")
    frame["quarter"] = "Q" + frame["year_month"].dt.quarter.astype(str)
    frame["year_month_label"] = frame["year_month"].dt.strftime("%Y-%m")
    frame.insert(0, "record_id", np.arange(1, len(frame) + 1, dtype="int64"))

    duplicate_keys = int(
        frame.duplicated(["model", "make", "year_month", "low_price", "high_price"]).sum()
    )
    if duplicate_keys:
        logger.warning("清洗后仍存在同价格配置重复记录：%s 条", duplicate_keys)
    else:
        logger.info("车型-厂商-月份-价格配置组合校验通过：无重复")

    frame = frame.sort_values(["year_month", "units_sold"], ascending=[True, False])
    logger.info(
        "清洗完成：%s 行；时间范围 %s 至 %s",
        len(frame),
        frame["year_month"].min().date(),
        frame["year_month"].max().date(),
    )
    return frame.reset_index(drop=True)


def safe_growth(current: pd.Series, prior: pd.Series) -> pd.Series:
    return np.where(prior > 0, current / prior - 1, np.nan)


def build_monthly(clean: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        clean.groupby("year_month", as_index=False)
        .agg(
            total_units=("units_sold", "sum"),
            model_count=("model", "nunique"),
            make_count=("make", "nunique"),
        )
        .sort_values("year_month")
    )
    ev = (
        clean.loc[clean["energy_type"].eq("新能源")]
        .groupby("year_month")["units_sold"]
        .sum()
    )
    domestic = (
        clean.loc[clean["brand_origin"].eq("中国品牌")]
        .groupby("year_month")["units_sold"]
        .sum()
    )
    monthly["ev_units"] = monthly["year_month"].map(ev).fillna(0).astype("int64")
    monthly["gasoline_units"] = monthly["total_units"] - monthly["ev_units"]
    monthly["domestic_units"] = monthly["year_month"].map(domestic).fillna(0).astype("int64")
    monthly["ev_share"] = monthly["ev_units"] / monthly["total_units"]
    monthly["domestic_share"] = monthly["domestic_units"] / monthly["total_units"]
    monthly["mom_growth"] = monthly["total_units"].pct_change()
    monthly["yoy_growth"] = monthly["total_units"].pct_change(12)
    monthly["year_month_label"] = monthly["year_month"].dt.strftime("%Y-%m")
    return monthly


def period_bounds(clean: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    current_end = clean["year_month"].max()
    current_start = current_end - pd.DateOffset(months=11)
    prior_end = current_start - pd.DateOffset(months=1)
    prior_start = prior_end - pd.DateOffset(months=11)
    return current_start, current_end, prior_start, prior_end


def period_comparison(clean: pd.DataFrame, group: str) -> pd.DataFrame:
    current_start, current_end, prior_start, prior_end = period_bounds(clean)
    current = clean[clean["year_month"].between(current_start, current_end)]
    prior = clean[clean["year_month"].between(prior_start, prior_end)]
    current_sales = current.groupby(group)["units_sold"].sum().rename("current_12m_units")
    prior_sales = prior.groupby(group)["units_sold"].sum().rename("prior_12m_units")
    result = pd.concat([current_sales, prior_sales], axis=1).fillna(0).reset_index()
    result["growth_rate"] = safe_growth(result["current_12m_units"], result["prior_12m_units"])
    result["current_share"] = result["current_12m_units"] / result["current_12m_units"].sum()
    return result.sort_values("current_12m_units", ascending=False)


def build_brand_summary(clean: pd.DataFrame) -> pd.DataFrame:
    summary = period_comparison(clean, "brand")
    current_start, current_end, _, _ = period_bounds(clean)
    current = clean[clean["year_month"].between(current_start, current_end)].copy()
    weighted_price = (
        (current["avg_price_k_rmb"] * current["units_sold"])
        .groupby(current["brand"])
        .sum()
        / current.groupby("brand")["units_sold"].sum()
    ).rename("weighted_avg_price_k_rmb")
    attributes = (
        clean.sort_values("year_month")
        .groupby("brand", as_index=False)
        .agg(
            brand_country=("brand_country", "last"),
            country_cn=("country_cn", "last"),
            brand_origin=("brand_origin", "last"),
            make_count=("make", "nunique"),
            model_count=("model", "nunique"),
        )
    )
    return (
        summary.merge(attributes, on="brand", how="left")
        .merge(weighted_price, on="brand", how="left")
    )


def build_model_summary(clean: pd.DataFrame) -> pd.DataFrame:
    summary = period_comparison(clean, "model")
    attributes = (
        clean.sort_values("year_month")
        .groupby("model", as_index=False)
        .agg(
            make=("make", "last"),
            brand=("brand", "last"),
            energy_type=("energy_type", "last"),
            body_type_cn=("body_type_cn", "last"),
            brand_origin=("brand_origin", "last"),
            avg_price_k_rmb=("avg_price_k_rmb", "median"),
        )
    )
    return summary.merge(attributes, on="model", how="left")


def build_segment_summary(clean: pd.DataFrame) -> pd.DataFrame:
    current_start, current_end, _, _ = period_bounds(clean)
    current = clean[clean["year_month"].between(current_start, current_end)]
    rows: list[pd.DataFrame] = []
    for dimension, label in (
        ("energy_type", "能源类型"),
        ("body_type_cn", "车型类别"),
        ("price_band", "价格带"),
        ("brand_origin", "品牌来源"),
    ):
        grouped = current.groupby(dimension, dropna=False)["units_sold"].sum().reset_index()
        grouped.columns = ["segment", "units_sold"]
        grouped["dimension"] = label
        grouped["share"] = grouped["units_sold"] / grouped["units_sold"].sum()
        rows.append(grouped[["dimension", "segment", "units_sold", "share"]])
    return pd.concat(rows, ignore_index=True)


def build_tableau_fact(
    clean: pd.DataFrame,
    monthly: pd.DataFrame,
    brands: pd.DataFrame,
    models: pd.DataFrame,
) -> pd.DataFrame:
    """输出单一明细粒度的 Tableau 事实表，不混合 KPI、品牌和车型粒度。"""
    fact = clean.copy()
    fact["new_energy_sales"] = np.where(
        fact["energy_type"].eq("新能源"), fact["units_sold"], 0
    ).astype("int64")
    fact["china_brand_sales"] = np.where(
        fact["brand_origin"].eq("中国品牌"), fact["units_sold"], 0
    ).astype("int64")

    month_metrics = monthly[
        ["year_month", "total_units", "mom_growth", "yoy_growth"]
    ].rename(
        columns={
            "total_units": "market_month_sales",
            "mom_growth": "market_mom_growth",
            "yoy_growth": "market_yoy_growth",
        }
    )
    fact = fact.merge(month_metrics, on="year_month", how="left", validate="many_to_one")

    brand_month = (
        fact.groupby(["year_month", "brand"], as_index=False)["units_sold"]
        .sum()
        .rename(columns={"units_sold": "brand_month_sales"})
        .sort_values(["brand", "year_month"])
    )
    brand_prior = brand_month[["year_month", "brand", "brand_month_sales"]].copy()
    brand_prior["year_month"] = brand_prior["year_month"] + pd.DateOffset(years=1)
    brand_prior = brand_prior.rename(columns={"brand_month_sales": "brand_prior_year_month_sales"})
    brand_month = brand_month.merge(
        brand_prior, on=["year_month", "brand"], how="left", validate="one_to_one"
    )
    brand_month["brand_yoy_growth"] = safe_growth(
        brand_month["brand_month_sales"], brand_month["brand_prior_year_month_sales"]
    )
    brand_month["brand_market_share"] = (
        brand_month["brand_month_sales"]
        / brand_month["year_month"].map(monthly.set_index("year_month")["total_units"])
    )
    fact = fact.merge(brand_month, on=["year_month", "brand"], how="left", validate="many_to_one")

    model_month = (
        fact.groupby(["year_month", "model"], as_index=False)["units_sold"]
        .sum()
        .rename(columns={"units_sold": "model_month_sales"})
        .sort_values(["model", "year_month"])
    )
    model_prior = model_month[["year_month", "model", "model_month_sales"]].copy()
    model_prior["year_month"] = model_prior["year_month"] + pd.DateOffset(years=1)
    model_prior = model_prior.rename(columns={"model_month_sales": "model_prior_year_month_sales"})
    model_month = model_month.merge(
        model_prior, on=["year_month", "model"], how="left", validate="one_to_one"
    )
    model_month["model_yoy_growth"] = safe_growth(
        model_month["model_month_sales"], model_month["model_prior_year_month_sales"]
    )
    model_month["model_market_share"] = (
        model_month["model_month_sales"]
        / model_month["year_month"].map(monthly.set_index("year_month")["total_units"])
    )
    fact = fact.merge(model_month, on=["year_month", "model"], how="left", validate="many_to_one")

    # 增长矩阵使用近 12 个月对比上一完整 12 个月的稳定口径。
    # 品牌保留销量 TOP15；车型先要求上期销量达到 10 万辆，再在稳定样本中取 TOP20。
    brand_metrics = brands[
        [
            "brand", "current_12m_units", "prior_12m_units", "growth_rate",
            "weighted_avg_price_k_rmb",
        ]
    ].copy()
    brand_metrics["brand_rank_12m"] = (
        brand_metrics["current_12m_units"].rank(method="first", ascending=False).astype("int64")
    )
    brand_metrics["brand_matrix_scope"] = np.where(
        brand_metrics["brand_rank_12m"].le(15)
        & brand_metrics["prior_12m_units"].ge(10_000)
        & brand_metrics["growth_rate"].notna(),
        "TOP15",
        "不展示",
    )
    brand_metrics = brand_metrics.rename(
        columns={
            "current_12m_units": "brand_current_12m_units",
            "prior_12m_units": "brand_prior_12m_units",
            "growth_rate": "brand_growth_12m",
            "weighted_avg_price_k_rmb": "brand_avg_price_12m",
        }
    )
    fact = fact.merge(brand_metrics, on="brand", how="left", validate="many_to_one")

    model_metrics = models[
        ["model", "current_12m_units", "prior_12m_units", "growth_rate"]
    ].copy()
    stable_model = model_metrics["prior_12m_units"].ge(100_000) & model_metrics["growth_rate"].notna()
    model_metrics["model_rank_12m"] = np.nan
    model_metrics.loc[stable_model, "model_rank_12m"] = model_metrics.loc[
        stable_model, "current_12m_units"
    ].rank(method="first", ascending=False)
    model_metrics["model_matrix_scope"] = np.where(
        model_metrics["model_rank_12m"].le(20), "TOP20", "不展示"
    )
    model_metrics = model_metrics.rename(
        columns={
            "current_12m_units": "model_current_12m_units",
            "prior_12m_units": "model_prior_12m_units",
            "growth_rate": "model_growth_12m",
        }
    )
    fact = fact.merge(model_metrics, on="model", how="left", validate="many_to_one")

    tableau_columns = [
        "record_id", "year_month", "year_month_label", "year", "quarter", "month",
        "brand_origin", "country_cn", "brand", "make", "model", "energy_type",
        "body_type_cn", "price_band", "units_sold", "new_energy_sales",
        "china_brand_sales", "avg_price_k_rmb", "market_month_sales",
        "market_mom_growth", "market_yoy_growth", "brand_month_sales",
        "brand_yoy_growth", "brand_market_share", "model_month_sales",
        "model_yoy_growth", "model_market_share", "brand_current_12m_units",
        "brand_prior_12m_units", "brand_growth_12m", "brand_rank_12m",
        "brand_avg_price_12m", "brand_matrix_scope", "model_current_12m_units",
        "model_prior_12m_units", "model_growth_12m", "model_rank_12m",
        "model_matrix_scope",
    ]
    return fact[tableau_columns].sort_values(
        ["year_month", "units_sold"], ascending=[True, False]
    )


def build_kpis(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    monthly: pd.DataFrame,
    brands: pd.DataFrame,
    models: pd.DataFrame,
) -> dict[str, object]:
    current_start, current_end, prior_start, prior_end = period_bounds(clean)
    current = clean[clean["year_month"].between(current_start, current_end)]
    prior = clean[clean["year_month"].between(prior_start, prior_end)]
    current_units = int(current["units_sold"].sum())
    prior_units = int(prior["units_sold"].sum())
    current_ev = int(current.loc[current["energy_type"].eq("新能源"), "units_sold"].sum())
    prior_ev = int(prior.loc[prior["energy_type"].eq("新能源"), "units_sold"].sum())
    current_domestic = int(current.loc[current["brand_origin"].eq("中国品牌"), "units_sold"].sum())
    prior_domestic = int(prior.loc[prior["brand_origin"].eq("中国品牌"), "units_sold"].sum())
    top5_share = float(brands.head(5)["current_share"].sum())
    latest_month = monthly.iloc[-1]
    return {
        "source_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "duplicates_removed": int(raw.duplicated().sum()),
        "missing_body_type_filled": int(raw["body_type"].isna().sum()),
        "start_month": clean["year_month"].min().strftime("%Y-%m"),
        "end_month": clean["year_month"].max().strftime("%Y-%m"),
        "current_period": f"{current_start:%Y-%m} 至 {current_end:%Y-%m}",
        "prior_period": f"{prior_start:%Y-%m} 至 {prior_end:%Y-%m}",
        "current_12m_units": current_units,
        "prior_12m_units": prior_units,
        "current_12m_growth": current_units / prior_units - 1,
        "current_ev_share": current_ev / current_units,
        "prior_ev_share": prior_ev / prior_units,
        "current_domestic_share": current_domestic / current_units,
        "prior_domestic_share": prior_domestic / prior_units,
        "top5_brand_share": top5_share,
        "top_brand": str(brands.iloc[0]["brand"]),
        "top_brand_units": int(brands.iloc[0]["current_12m_units"]),
        "top_model": str(models.iloc[0]["model"]),
        "top_model_units": int(models.iloc[0]["current_12m_units"]),
        "latest_month": str(latest_month["year_month_label"]),
        "latest_month_units": int(latest_month["total_units"]),
        "latest_month_yoy": float(latest_month["yoy_growth"]),
    }


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def write_conclusions(kpis: dict[str, object], brands: pd.DataFrame, models: pd.DataFrame) -> str:
    comparable_brands = brands.loc[
        (brands["prior_12m_units"] >= 100_000)
        & (brands["current_12m_units"] >= 100_000)
        & brands["growth_rate"].notna()
    ]
    high_growth = comparable_brands.loc[
        comparable_brands["growth_rate"].notna()
    ].sort_values("growth_rate", ascending=False).iloc[0]
    declining = comparable_brands.sort_values("growth_rate").iloc[0]
    model_growth = models.loc[
        (models["prior_12m_units"] >= 10_000)
        & (models["current_12m_units"] >= 50_000)
        & models["growth_rate"].notna()
    ].sort_values("growth_rate", ascending=False).iloc[0]

    text = f"""# 分析结论

分析窗口采用最近完整滚动 12 个月（{kpis['current_period']}），并与上一滚动 12 个月（{kpis['prior_period']}）比较。

1. **总量增长放缓后，竞争重点转向结构变化。** 最近 12 个月样本销量为 {int(kpis['current_12m_units']):,} 辆，同比 {format_percent(float(kpis['current_12m_growth']))}。市场规模仍在变化，但厂商不能只依赖行业自然增长，需要通过能源结构、产品组合和重点车型提升份额。
2. **新能源已成为最明确的结构性增长来源。** 新能源销量占比由 {format_percent(float(kpis['prior_ev_share']))} 提升至 {format_percent(float(kpis['current_ev_share']))}，增加 {(float(kpis['current_ev_share'])-float(kpis['prior_ev_share']))*100:.1f} 个百分点。产品投放和渠道资源应继续向新能源车型倾斜，同时单独监控燃油车库存与长尾车型退坡风险。
3. **中国品牌份额继续扩大。** 中国品牌份额由 {format_percent(float(kpis['prior_domestic_share']))} 提升至 {format_percent(float(kpis['current_domestic_share']))}。自主品牌的竞争优势已从价格带延伸到更广车型区间，海外品牌需要通过电动化产品和本地化定价应对份额压力。
4. **品牌表现出现明显分化。** 最近 12 个月销量最高品牌为 {kpis['top_brand']}，前五品牌合计占 {format_percent(float(kpis['top5_brand_share']))}；在前后两个周期销量均不少于 10 万辆的品牌中，{high_growth['brand']} 增长最快（{format_percent(float(high_growth['growth_rate']))}），而 {declining['brand']} 降幅最大（{format_percent(float(declining['growth_rate']))}）。资源配置应同时考虑规模和增速，避免只按绝对销量排名。
5. **爆款车型仍能显著拉动品牌增长。** 最近 12 个月销量最高车型为 {kpis['top_model']}（{int(kpis['top_model_units']):,} 辆）；有一定历史销量基础的车型中，{model_growth['model']} 增长最快。建议以“核心走量车型 + 新增量车型 + 衰退车型”建立产品组合看板，分别制定保供、扩量和退出策略。

> 数据限制：数据为公开网站整理的车型月度销量，不包含企业成本、利润、渠道、库存和订单明细；因此结论聚焦市场销量结构，不能直接解释企业利润或员工绩效。
"""
    (REPORT_DIR / "analysis_conclusions.md").write_text(text, encoding="utf-8")
    return text


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def create_figures(
    clean: pd.DataFrame,
    monthly: pd.DataFrame,
    brands: pd.DataFrame,
    models: pd.DataFrame,
    segments: pd.DataFrame,
    kpis: dict[str, object],
) -> None:
    setup_chinese_font()
    colors = {"blue": "#1F4E78", "orange": "#F39C12", "green": "#2E8B57", "gray": "#B8C2CC"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(monthly["year_month"], monthly["total_units"], color=colors["blue"], linewidth=2)
    ax.set_title("月度汽车销量趋势")
    ax.set_ylabel("销量（辆）")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    ax.plot(monthly["year_month"], monthly["ev_share"] * 100, color=colors["green"], linewidth=2)
    ax.set_title("新能源销量占比")
    ax.set_ylabel("占比（%）")
    ax.grid(axis="y", alpha=0.2)

    top_brands = brands.head(10).sort_values("current_12m_units")
    ax = axes[1, 0]
    ax.barh(top_brands["brand"], top_brands["current_12m_units"], color=colors["blue"])
    ax.set_title("最近12个月品牌销量TOP10")
    ax.set_xlabel("销量（辆）")

    body = segments.loc[segments["dimension"].eq("车型类别")].sort_values("units_sold")
    ax = axes[1, 1]
    ax.barh(body["segment"], body["units_sold"], color=colors["orange"])
    ax.set_title("最近12个月车型类别结构")
    ax.set_xlabel("销量（辆）")
    fig.suptitle("中国汽车市场销量分析", fontsize=18, fontweight="bold")
    fig.savefig(FIGURE_DIR / "analysis_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    plot_data = brands.loc[
        (brands["prior_12m_units"] >= 5_000)
        & brands["growth_rate"].between(-0.8, 3.0, inclusive="both")
    ].copy()
    ax.scatter(
        plot_data["current_share"] * 100,
        plot_data["growth_rate"] * 100,
        s=np.sqrt(plot_data["current_12m_units"]) * 5,
        c=np.where(plot_data["brand_origin"].eq("中国品牌"), colors["orange"], colors["blue"]),
        alpha=0.7,
        edgecolor="white",
    )
    for row in plot_data.nlargest(12, "current_12m_units").itertuples(index=False):
        ax.annotate(row.brand, (row.current_share * 100, row.growth_rate * 100), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_xlabel("最近12个月市场份额（%）")
    ax.set_ylabel("同比增长率（%）")
    ax.set_title("品牌份额—增长矩阵")
    ax.grid(alpha=0.2)
    fig.savefig(FIGURE_DIR / "brand_growth_share_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("中国汽车市场销量分析｜Python分析预览", fontsize=20, fontweight="bold")

    axes[0, 0].text(0.02, 0.72, f"{int(kpis['current_12m_units']):,}", fontsize=28, fontweight="bold", color=colors["blue"])
    axes[0, 0].text(0.02, 0.52, "最近12个月销量（辆）", fontsize=12)
    axes[0, 0].text(0.02, 0.28, f"同比 {format_percent(float(kpis['current_12m_growth']))}", fontsize=18, color=colors["green"] if float(kpis['current_12m_growth']) >= 0 else "#C0392B")
    axes[0, 0].axis("off")

    axes[0, 1].plot(monthly["year_month"], monthly["total_units"], color=colors["blue"], linewidth=2)
    axes[0, 1].set_title("月度销量")
    axes[0, 1].grid(axis="y", alpha=0.2)

    axes[0, 2].plot(monthly["year_month"], monthly["ev_share"] * 100, color=colors["green"], linewidth=2)
    axes[0, 2].set_title("新能源占比")
    axes[0, 2].set_ylabel("%")
    axes[0, 2].grid(axis="y", alpha=0.2)

    top10 = brands.head(10).sort_values("current_12m_units")
    axes[1, 0].barh(top10["brand"], top10["current_12m_units"], color=colors["blue"])
    axes[1, 0].set_title("品牌销量TOP10")

    price = segments.loc[segments["dimension"].eq("价格带")].copy()
    price_order = ["10万元以下", "10–15万元", "15–20万元", "20–30万元", "30万元以上", "价格缺失"]
    price["order"] = price["segment"].map({v: i for i, v in enumerate(price_order)})
    price = price.sort_values("order")
    axes[1, 1].bar(price["segment"], price["units_sold"], color=colors["orange"])
    axes[1, 1].set_title("价格带销量结构")
    axes[1, 1].tick_params(axis="x", rotation=30)

    top_models = models.head(10).sort_values("current_12m_units")
    axes[1, 2].barh(top_models["model"], top_models["current_12m_units"], color=colors["green"])
    axes[1, 2].set_title("车型销量TOP10")

    for axis in axes.flat:
        axis.tick_params(labelsize=9)
    fig.savefig(FIGURE_DIR / "dashboard_preview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_source_metadata(log_path: Path) -> None:
    stat = RAW_FILE.stat()
    metadata = {
        "dataset_name": "China Automobile Monthly Sales Data (2018-2024.4)",
        "data_type": "真实公开整理数据，非模拟生成",
        "source_page": SOURCE_PAGE,
        "download_mirror": SOURCE_RAW_MIRROR,
        "license": "MIT",
        "raw_file": str(RAW_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "raw_file_size_bytes": stat.st_size,
        "raw_file_sha256": sha256(RAW_FILE),
        "local_file_timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "pipeline_run_log": "run_logs/latest_successful_run.log",
    }
    (PROJECT_ROOT / "source_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    ensure_directories()
    logger = configure_logging()
    logger.info("项目开始运行")
    logger.info("Python：%s", platform.python_version())
    logger.info("pandas：%s；NumPy：%s", pd.__version__, np.__version__)

    raw = load_raw(logger)
    clean = clean_and_engineer(raw, logger)
    monthly = build_monthly(clean)
    brands = build_brand_summary(clean)
    models = build_model_summary(clean)
    tableau_fact = build_tableau_fact(clean, monthly, brands, models)
    kpis = build_kpis(raw, clean, monthly, brands, models)

    clean_export = clean.copy()
    clean_export["year_month"] = clean_export["year_month"].dt.strftime("%Y-%m-%d")
    clean_export.to_csv(PROCESSED_DIR / "china_auto_sales_clean.csv", index=False, encoding="utf-8-sig")
    tableau_fact_export = tableau_fact.copy()
    tableau_fact_export["year_month"] = tableau_fact_export["year_month"].dt.strftime("%Y-%m-%d")
    tableau_fact_export.to_csv(
        TABLEAU_DIR / "china_auto_sales_tableau.csv", index=False, encoding="utf-8-sig"
    )
    logger.info("输出清洗数据：%s", PROCESSED_DIR / "china_auto_sales_clean.csv")
    logger.info(
        "输出 Tableau 单粒度事实表：%s（%s 行，%s 列）",
        TABLEAU_DIR / "china_auto_sales_tableau.csv",
        f"{len(tableau_fact):,}",
        len(tableau_fact.columns),
    )
    logger.info("最近12个月销量：%s；同比：%s", f"{int(kpis['current_12m_units']):,}", format_percent(float(kpis["current_12m_growth"])))
    logger.info("项目运行完成")


if __name__ == "__main__":
    main()
