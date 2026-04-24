import io
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

fig = px.line(
    plot_df,
    x=x_col,
    y=y_to_plot,
    color=color_col,
    title=f"{y_col} trend",
)
fig.update_layout(legend_title_text="Series", margin=dict(l=20, r=20, t=50, b=20))
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
