"""Streamlit UI launched by ``dashboard.MetricsDashboard``, not as a trainer."""

from __future__ import annotations

import csv
import io
import json
import os
from collections import deque
from urllib.error import URLError
from urllib.request import Request, urlopen

import plotly.graph_objects as go
import streamlit as st

from dashboard import DashboardSnapshot


def fetch_snapshot() -> DashboardSnapshot:
    request = Request(
        f"{os.environ['SHOOTER_DASHBOARD_URL']}/snapshot",
        headers={"Authorization": f"Bearer {os.environ['SHOOTER_DASHBOARD_TOKEN']}"},
    )
    with urlopen(request, timeout=2) as response:
        return json.load(response)


def request_stop() -> None:
    request = Request(
        f"{os.environ['SHOOTER_DASHBOARD_URL']}/stop",
        headers={"Authorization": f"Bearer {os.environ['SHOOTER_DASHBOARD_TOKEN']}"},
        data=b"",
        method="POST",
    )
    with urlopen(request, timeout=2) as response:
        response.read()


def export_csv(snapshot: DashboardSnapshot) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["metric", "environment_step", "value"])
    for key, points in snapshot["history"].items():
        writer.writerows((key, step, value) for step, value in points)
    return output.getvalue()


def smoothed(points: list[tuple[int, float | None]], window: int) -> list[float | None]:
    values: deque[float | None] = deque()
    total, count = 0.0, 0
    result: list[float | None] = []
    for _, value in points:
        values.append(value)
        if value is not None:
            total += value
            count += 1
        if len(values) > window:
            expired = values.popleft()
            if expired is not None:
                total -= expired
                count -= 1
        result.append(total / count if count and value is not None else None)
    return result


def metric_figure(
    snapshot: DashboardSnapshot, key: str, smoothing: int, show_raw: bool
) -> go.Figure:
    points = snapshot["history"][key]
    steps = [step for step, _ in points]
    figure = go.Figure()
    if show_raw:
        figure.add_trace(go.Scatter(
            x=steps, y=[value for _, value in points], name="Raw",
            mode="lines", line={"color": "rgba(54,214,176,0.3)", "width": 1},
        ))
    figure.add_trace(go.Scatter(
        x=steps, y=smoothed(points, smoothing), name="Smoothed",
        mode="lines", line={"color": "#36D6B0", "width": 2.5},
    ))
    if key in snapshot["reference_lines"]:
        figure.add_hline(
            y=snapshot["reference_lines"][key], line_dash="dash",
            line_color="#FFB454", annotation_text="Promotion threshold",
        )
    figure.update_layout(
        template="plotly_dark", height=265,
        margin={"l": 8, "r": 8, "t": 20, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Environment steps", hovermode="x unified",
        legend={"orientation": "h", "y": 1.15},
        uirevision=key,
    )
    figure.update_xaxes(gridcolor="rgba(148,163,184,0.1)")
    figure.update_yaxes(gridcolor="rgba(148,163,184,0.1)")
    return figure


def render_charts(snapshot: DashboardSnapshot, keys: list[str], smoothing: int, show_raw: bool) -> None:
    columns = st.columns(min(snapshot["columns"], 4))
    for index, key in enumerate(keys):
        with columns[index % len(columns)]:
            with st.container(border=True):
                st.markdown(f"**{snapshot['metrics'][key]}**")
                if snapshot["history"][key]:
                    st.plotly_chart(
                        metric_figure(snapshot, key, smoothing, show_raw),
                        width="stretch", key=f"chart_{key}",
                        config={"displaylogo": False},
                    )
                else:
                    st.caption("Waiting for data...")


@st.fragment(run_every=1.0)
def live_dashboard() -> None:
    try:
        snapshot = fetch_snapshot()
    except (URLError, TimeoutError, ConnectionError) as error:
        st.warning("Training has disconnected. The dashboard server closes when training ends.")
        st.caption(f"Connection details: {error}")
        return

    st.caption("ADAPTIVE SHOOTER / TRAINING TELEMETRY")
    title, control = st.columns([4, 1])
    with title:
        st.title(snapshot["title"])
        st.caption("Live local telemetry  |  Updates every second  |  Drag to zoom, double-click to reset")
    with control:
        stopping = snapshot["stop_requested"]
        st.markdown("**STOPPING**" if stopping else "**LIVE**")
        if st.button("Stop after episode", type="primary", disabled=stopping, width="stretch"):
            try:
                request_stop()
            except (URLError, TimeoutError, ConnectionError) as error:
                st.error(f"Could not request a graceful stop: {error}")
            else:
                st.rerun()
    if stopping:
        st.info("Finishing current episodes, then saving the checkpoint. Training will close the dashboard.")

    info = dict(line.split(maxsplit=1) for line in snapshot["info"].splitlines() if len(line.split()) >= 2)
    cards = st.columns(4)
    for column, label, value in zip(
        cards, ["Environment steps", "Episodes", "PPO updates", "Steps / second"],
        [info.get("steps", "--"), info.get("episodes", "--"), info.get("updates", "--"), info.get("SPS", "--")],
    ):
        column.metric(label, value, border=True)
    kpis = [("mean_reward", "Mean reward"), ("mean_kills", "Mean kills"), ("hit_rate", "Hit rate")]
    if "death_rate" in snapshot["metrics"]:
        kpis.append(("death_rate", "Death rate"))
    else:
        kpis.append(("stage", "Curriculum stage"))
    for column, (key, label) in zip(st.columns(4), kpis):
        points = snapshot["history"].get(key, [])
        latest = points[-1][1] if points else None
        previous = points[-2][1] if len(points) > 1 else None
        value = "--" if latest is None else (
            f"{latest:.1%}" if key.endswith("rate") else f"{latest:,.2f}"
        )
        delta = f"{latest - previous:+.3f}" if latest is not None and previous is not None else None
        column.metric(label, value, delta=delta, delta_color="inverse" if key == "death_rate" else "normal", border=True)

    with st.expander("Chart controls and run details"):
        left, right = st.columns(2)
        with left:
            smoothing = st.slider("Smoothing window (points)", 1, 100, min(snapshot["smoothing"], 100))
            show_raw = st.checkbox("Show raw values", value=True)
            st.caption("Metrics without a value (NaN/Inf) are shown as gaps, not zeros.")
        with right:
            st.text(snapshot["info"])
            st.download_button(
                "Download metrics CSV", export_csv(snapshot),
                file_name="training_metrics.csv", mime="text/csv",
            )
            st.caption("Export contains raw values retained in memory (up to 5,000 points per metric by default).")

    combat_keys = [key for key in ("mean_damage_taken", "death_rate") if key in snapshot["metrics"]]
    episode_keys = [
        key for key in snapshot["metrics"]
        if (key.startswith("mean_") or key in ("stage", "hit_rate")) and key not in combat_keys
    ]
    optimization_keys = [key for key in snapshot["metrics"] if key not in episode_keys + combat_keys]
    tabs = st.tabs(["Agent performance", "PPO optimization"] + (["Combat survival"] if combat_keys else []))
    performance, optimization = tabs[:2]
    with performance:
        render_charts(snapshot, episode_keys, smoothing, show_raw)
    with optimization:
        render_charts(snapshot, optimization_keys, smoothing, show_raw)
    if combat_keys:
        with tabs[2]:
            render_charts(snapshot, combat_keys, smoothing, show_raw)


def main() -> None:
    st.set_page_config(page_title="Adaptive Shooter | Training", page_icon=":material/monitoring:", layout="wide")
    st.markdown(
        "<style>"
        ".block-container {padding-top:2rem; padding-bottom:2rem; max-width:1600px;}"
        "[data-testid='stMetricValue'] {font-variant-numeric:tabular-nums;}"
        "h1 {letter-spacing:-0.04em;}"
        "</style>",
        unsafe_allow_html=True,
    )
    if not os.environ.get("SHOOTER_DASHBOARD_URL") or not os.environ.get("SHOOTER_DASHBOARD_TOKEN"):
        st.error("Start training_random_map.py to launch a connected dashboard.")
        st.stop()
    live_dashboard()


if __name__ == "__main__":
    main()
