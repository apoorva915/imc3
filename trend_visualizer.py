import io
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Trading CSV Trend Visualizer", layout="wide")
st.title("Trading Contest CSV Trend Visualizer")
st.caption("Simple, clean trend analysis for one or many CSV files.")


def detect_separator(file_bytes: bytes) -> str:
    sample = file_bytes[:5000].decode("utf-8", errors="ignore")
    return ";" if sample.count(";") > sample.count(",") else ","


def load_csv_from_bytes(name: str, file_bytes: bytes) -> pd.DataFrame:
    sep = detect_separator(file_bytes)
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    df.columns = [c.strip() for c in df.columns]
    df["__source__"] = name
    return df


def add_x_axis_column(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        return df
    if "day" in df.columns and "timestamp" in df.columns:
        return df
    df = df.copy()
    df["row_index"] = range(len(df))
    return df


def candidate_x_columns(df: pd.DataFrame) -> list[str]:
    preferred = [c for c in ["day", "timestamp", "row_index"] if c in df.columns]
    if preferred:
        return preferred
    return df.select_dtypes(include=["number"]).columns.tolist()[:5]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include=["number"]).columns if c != "row_index"]


def parse_vev_strike(symbol: str) -> float | None:
    match = re.match(r"^VEV_(\d+)$", str(symbol))
    if not match:
        return None
    return float(match.group(1))


workspace = Path.cwd()
local_csvs = sorted(workspace.glob("*.csv"))

with st.sidebar:
    st.header("Data Selection")
    picked_local = st.multiselect(
        "Choose local CSV files",
        options=[p.name for p in local_csvs],
        default=[p.name for p in local_csvs[:2]],
    )
    uploaded_files = st.file_uploader(
        "Or upload one/multiple CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

loaded_frames: dict[str, pd.DataFrame] = {}

for file_name in picked_local:
    path = workspace / file_name
    file_bytes = path.read_bytes()
    try:
        loaded_frames[file_name] = load_csv_from_bytes(file_name, file_bytes)
    except Exception as exc:
        st.warning(f"Could not read {file_name}: {exc}")

if uploaded_files:
    for up in uploaded_files:
        try:
            loaded_frames[up.name] = load_csv_from_bytes(up.name, up.getvalue())
        except Exception as exc:
            st.warning(f"Could not read {up.name}: {exc}")

if not loaded_frames:
    st.info("Select at least one CSV from sidebar to begin.")
    st.stop()

with st.sidebar:
    st.header("Include / Exclude")
    active_sources = st.multiselect(
        "Show data from files",
        options=list(loaded_frames.keys()),
        default=list(loaded_frames.keys()),
    )

if not active_sources:
    st.info("No files selected in 'Show data from files'.")
    st.stop()

frames = []
for source in active_sources:
    frame = add_x_axis_column(loaded_frames[source])
    frames.append(frame)

combined = pd.concat(frames, ignore_index=True)

x_options = candidate_x_columns(combined)
y_options = numeric_columns(combined)

if not x_options or not y_options:
    st.error("Could not find enough numeric columns to plot.")
    st.stop()

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    x_col = st.selectbox("X-axis", options=x_options, index=0)
with col2:
    y_col = st.selectbox(
        "Metric (Y-axis)",
        options=[c for c in y_options if c not in {"day", "timestamp"}] or y_options,
        index=0,
    )
with col3:
    smooth_window = st.slider("Smoothing (moving average)", 1, 50, 1)

compare_metrics = st.multiselect(
    "Compare extra metrics on same chart (optional)",
    options=[c for c in y_options if c not in {"day", "timestamp", y_col}],
    default=[],
    help="Example: pick bid/ask along with mid_price for the selected product.",
)

aggregate_same_x = st.checkbox(
    "Aggregate same timestamp values (mean) for cleaner line",
    value=True,
    help="Removes line thickening when many rows share the same timestamp.",
)

normalize_series = st.checkbox(
    "Normalize Y for easier comparison (start at 100)",
    value=False,
    help="Useful when products have very different price scales.",
)

category_candidates = [c for c in ["product", "symbol"] if c in combined.columns]
group_by = st.selectbox("Split lines by", options=["None", "__source__"] + category_candidates)

plot_df = combined.copy()
if group_by != "None":
    unique_vals = sorted(plot_df[group_by].dropna().astype(str).unique().tolist())
    selected_vals = st.multiselect(
        f"Filter {group_by} values",
        options=unique_vals,
        default=unique_vals[: min(12, len(unique_vals))],
    )
    if selected_vals:
        plot_df = plot_df[plot_df[group_by].astype(str).isin(selected_vals)]

plot_df = plot_df.sort_values(by=[x_col])

if smooth_window > 1:
    group_keys = ["__source__"]
    if group_by != "None":
        group_keys.append(group_by)
    plot_df[f"{y_col}_smoothed"] = (
        plot_df.groupby(group_keys, dropna=False)[y_col]
        .transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
    )
    y_to_plot = f"{y_col}_smoothed"
else:
    y_to_plot = y_col

color_col = None if group_by == "None" else group_by
if group_by == "None" and len(active_sources) > 1:
    color_col = "__source__"

if normalize_series:
    norm_group = ["__source__"]
    if group_by != "None":
        norm_group.append(group_by)
    normalized_col = f"{y_to_plot}_normalized"
    first_vals = plot_df.groupby(norm_group, dropna=False)[y_to_plot].transform("first")
    plot_df[normalized_col] = (plot_df[y_to_plot] / first_vals.replace(0, pd.NA)) * 100
    y_to_plot = normalized_col

metric_cols = [y_col] + compare_metrics
if len(metric_cols) > 1:
    melted_df = plot_df.melt(
        id_vars=[c for c in [x_col, "__source__", group_by] if c != "None"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    if aggregate_same_x:
        agg_cols = [x_col, "metric"]
        if color_col:
            agg_cols.append(color_col)
        melted_df = (
            melted_df.groupby(agg_cols, dropna=False, as_index=False)["value"]
            .mean()
            .sort_values(by=[x_col])
        )

    if color_col:
        melted_df["series"] = melted_df[color_col].astype(str) + " | " + melted_df["metric"].astype(str)
        color_for_plot = "series"
    else:
        color_for_plot = "metric"

    fig = px.line(
        melted_df,
        x=x_col,
        y="value",
        color=color_for_plot,
        title=f"{', '.join(metric_cols)} trend",
        render_mode="webgl",
    )
else:
    single_df = plot_df.copy()
    if aggregate_same_x:
        agg_cols = [x_col]
        if color_col:
            agg_cols.append(color_col)
        single_df = (
            single_df.groupby(agg_cols, dropna=False, as_index=False)[y_to_plot]
            .mean()
            .sort_values(by=[x_col])
        )

    fig = px.line(
        single_df,
        x=x_col,
        y=y_to_plot,
        color=color_col,
        title=f"{y_col} trend",
        render_mode="webgl",
    )

fig.update_traces(line={"width": 1.3})
fig.update_layout(legend_title_text="Series", margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Quick Summary")
summary_keys = ["count", "mean", "std", "min", "max"]
if group_by == "None":
    summary_df = plot_df[[y_col]].describe().loc[summary_keys].T
else:
    summary_df = (
        plot_df.groupby(group_by, dropna=False)[y_col]
        .describe()
        .reset_index()[[group_by] + summary_keys]
        .sort_values(by="count", ascending=False)
    )
st.dataframe(summary_df, use_container_width=True)

with st.expander("Distribution view (simple box plot)"):
    if group_by == "None":
        dist_fig = px.box(plot_df, y=y_col, points=False, title=f"{y_col} distribution")
    else:
        dist_fig = px.box(plot_df, x=group_by, y=y_col, points=False, title=f"{y_col} distribution by {group_by}")
    dist_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(dist_fig, use_container_width=True)

with st.expander("Preview active data"):
    st.dataframe(plot_df.head(300), use_container_width=True)


st.divider()
st.header("Round 4 Insights")

trades_sources = [name for name, df in loaded_frames.items() if {"buyer", "seller", "symbol", "price", "quantity"}.issubset(df.columns)]
prices_sources = [name for name, df in loaded_frames.items() if {"product", "mid_price", "timestamp"}.issubset(df.columns)]

tab_counterparty, tab_vev = st.tabs(["Counterparty (Mark IDs)", "VEV Analysis"])

with tab_counterparty:
    if not trades_sources:
        st.info("Load at least one trades CSV to see counterparty analysis.")
    else:
        trades_all = pd.concat([loaded_frames[s].copy() for s in trades_sources], ignore_index=True)
        trades_all["price"] = pd.to_numeric(trades_all["price"], errors="coerce")
        trades_all["quantity"] = pd.to_numeric(trades_all["quantity"], errors="coerce")
        trades_all = trades_all.dropna(subset=["price", "quantity"])
        trades_all["notional"] = trades_all["price"] * trades_all["quantity"]
        trades_all["symbol"] = trades_all["symbol"].astype(str)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            cp_product = st.selectbox(
                "Product / Symbol",
                options=["All"] + sorted(trades_all["symbol"].unique().tolist()),
                key="cp_product",
            )
        with col_b:
            cp_side = st.selectbox("Side", options=["Both", "Buyer only", "Seller only"], key="cp_side")
        with col_c:
            top_n = st.slider("Show top N counterparties", 5, 30, 12, key="cp_top_n")

        cp_df = trades_all.copy()
        if cp_product != "All":
            cp_df = cp_df[cp_df["symbol"] == cp_product]

        if cp_side == "Buyer only":
            cp_summary = (
                cp_df.groupby("buyer", dropna=False)
                .agg(trades=("quantity", "size"), quantity=("quantity", "sum"), notional=("notional", "sum"))
                .reset_index()
                .rename(columns={"buyer": "counterparty"})
                .sort_values("notional", ascending=False)
                .head(top_n)
            )
        elif cp_side == "Seller only":
            cp_summary = (
                cp_df.groupby("seller", dropna=False)
                .agg(trades=("quantity", "size"), quantity=("quantity", "sum"), notional=("notional", "sum"))
                .reset_index()
                .rename(columns={"seller": "counterparty"})
                .sort_values("notional", ascending=False)
                .head(top_n)
            )
        else:
            buy_side = (
                cp_df.groupby("buyer", dropna=False)["quantity"]
                .sum()
                .reset_index()
                .rename(columns={"buyer": "counterparty", "quantity": "buy_qty"})
            )
            sell_side = (
                cp_df.groupby("seller", dropna=False)["quantity"]
                .sum()
                .reset_index()
                .rename(columns={"seller": "counterparty", "quantity": "sell_qty"})
            )
            cp_summary = buy_side.merge(sell_side, on="counterparty", how="outer").fillna(0)
            cp_summary["net_qty"] = cp_summary["buy_qty"] - cp_summary["sell_qty"]
            cp_summary["abs_net"] = cp_summary["net_qty"].abs()
            cp_summary = cp_summary.sort_values("abs_net", ascending=False).head(top_n)

        if cp_summary.empty:
            st.info("No matching rows for this filter.")
        else:
            if cp_side == "Both":
                fig_cp = px.bar(cp_summary, x="counterparty", y="net_qty", title="Net traded quantity by counterparty")
                fig_cp.update_layout(xaxis_title="", yaxis_title="Net quantity (buy - sell)")
                st.plotly_chart(fig_cp, use_container_width=True)
            else:
                fig_cp = px.bar(cp_summary, x="counterparty", y="notional", title="Counterparty notional activity")
                fig_cp.update_layout(xaxis_title="", yaxis_title="Notional")
                st.plotly_chart(fig_cp, use_container_width=True)
            st.dataframe(cp_summary, use_container_width=True)

with tab_vev:
    if not prices_sources:
        st.info("Load at least one prices CSV for VEV analysis.")
    else:
        prices_all = pd.concat([loaded_frames[s].copy() for s in prices_sources], ignore_index=True)
        prices_all["product"] = prices_all["product"].astype(str)
        vev_prices = prices_all[prices_all["product"].str.startswith("VEV_")].copy()

        if vev_prices.empty:
            st.info("No VEV products found in selected prices files.")
        else:
            vev_prices["strike"] = vev_prices["product"].apply(parse_vev_strike)
            underlying = prices_all[prices_all["product"] == "VELVETFRUIT_EXTRACT"][["day", "timestamp", "mid_price"]].rename(
                columns={"mid_price": "underlying_mid"}
            )
            vev_joined = vev_prices.merge(underlying, on=["day", "timestamp"], how="left")
            vev_joined["intrinsic"] = (vev_joined["underlying_mid"] - vev_joined["strike"]).clip(lower=0)
            vev_joined["premium_over_intrinsic"] = vev_joined["mid_price"] - vev_joined["intrinsic"]

            vev_symbols = sorted(vev_joined["product"].unique().tolist())
            selected_vevs = st.multiselect(
                "VEVs to compare",
                options=vev_symbols,
                default=vev_symbols[: min(4, len(vev_symbols))],
                key="vev_symbols",
            )
            vev_metric = st.selectbox(
                "VEV metric",
                options=["mid_price", "intrinsic", "premium_over_intrinsic"],
                index=2,
                key="vev_metric",
            )

            vev_plot = vev_joined[vev_joined["product"].isin(selected_vevs)].sort_values(["day", "timestamp"])
            if vev_plot.empty:
                st.info("Select at least one VEV.")
            else:
                fig_vev = px.line(
                    vev_plot,
                    x="timestamp",
                    y=vev_metric,
                    color="product",
                    facet_row="day",
                    title=f"VEV {vev_metric} by day",
                    render_mode="webgl",
                )
                fig_vev.update_traces(line={"width": 1.2})
                fig_vev.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_vev, use_container_width=True)

                tte_note = "TTE starts at 7 on day 1 and drops by 1 each day."
                st.caption(tte_note)
