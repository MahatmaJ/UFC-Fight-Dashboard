"""
UFC Fights Dashboard (1993-2025)
--------------------------------
An interactive Plotly Dash app exploring:
  1. Fights per year, broken down by win method
  2. Significant strike accuracy by weight class (winners vs. losers)
  3. Control time vs. win method, and win rate by control-time advantage
  4. Win rate by stance (overall, and by weight class)
  5. Fighter search / head-to-head comparison

Layout: dark theme, header banner, KPI stat cards, chart cards - restyled
with a UFC-inspired black / red / gold palette. No UFC trademarks or logos
are used - only the color language.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:8050 in your browser.

Data assumption: following the UFCStats.com convention that this dataset
was scraped from, `fighter_1` is the winner of the bout and `fighter_2`
is the loser, except for rows where `method` is "Overturned" or
"Could Not Continue" (no meaningful winner) - those rows are excluded
from any analysis that depends on knowing who won, but still counted in
the raw "fights per year" totals.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, ctx

# --------------------------------------------------------------------------
# 1. Load & prepare data
# --------------------------------------------------------------------------

DATA_PATH = "clean_ufc_dataset.csv"
df = pd.read_csv(DATA_PATH)

df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
df["year"] = df["event_date"].dt.year

METHOD_MAP = {
    "Decision - Unanimous": "Decision",
    "Decision - Split": "Decision",
    "Decision - Majority": "Decision",
    "KO/TKO": "KO/TKO",
    "TKO - Doctor's Stoppage": "KO/TKO",
    "Submission": "Submission",
    "DQ": "DQ",
    "Overturned": "Overturned/No Winner",
    "Could Not Continue": "Overturned/No Winner",
    "Other": "Other",
}
df["method_group"] = df["method"].map(METHOD_MAP).fillna("Other")

NO_WINNER = {"Overturned/No Winner"}
df["has_winner"] = ~df["method_group"].isin(NO_WINNER)


def ctrl_to_seconds(val):
    """Convert 'MM:SS' control-time strings to seconds."""
    if pd.isna(val):
        return None
    try:
        m, s = str(val).split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


df["f1_ctrl_sec"] = df["f1_Ctrl"].apply(ctrl_to_seconds)
df["f2_ctrl_sec"] = df["f2_Ctrl"].apply(ctrl_to_seconds)
df["winner_ctrl_sec"] = df["f1_ctrl_sec"]

YEAR_MIN, YEAR_MAX = int(df["year"].min()), int(df["year"].max())

WEIGHT_ORDER = [
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
    "Middleweight", "Light Heavyweight", "Heavyweight", "Super Heavyweight",
    "Catch Weight", "Open Weight",
]
WEIGHT_CLASSES = [w for w in WEIGHT_ORDER if w in df["weight_class"].dropna().unique()]

METHOD_ORDER = ["Decision", "KO/TKO", "Submission", "DQ", "Other", "Overturned/No Winner"]
MAIN_STANCES = ["Orthodox", "Southpaw", "Switch"]

# ---- Build one unified long-format table: one row per fighter per fight ----
# This underlies strike accuracy, control-time buckets, stance win rate, and
# the fighter search/comparison tool.

def build_perspective(df, mine, theirs):
    """mine/theirs = 'f1' or 'f2'; returns this fighter's own-perspective rows."""
    prefix_mine, prefix_theirs = f"{mine}_", f"{theirs}_"
    out = pd.DataFrame({
        "fighter": df[f"fighter_{mine[1]}"],
        "opponent": df[f"fighter_{theirs[1]}"],
        "weight_class": df["weight_class"],
        "year": df["year"],
        "method_group": df["method_group"],
        "has_winner": df["has_winner"],
        "stance": df[f"{prefix_mine}Stance"],
        "won": 1 if mine == "f1" else 0,
        "sig_landed": df[f"{prefix_mine}Sig_str_landed"],
        "sig_attempted": df[f"{prefix_mine}Sig_str_attempted"],
        "td_landed": df[f"{prefix_mine}Td_landed"],
        "td_attempted": df[f"{prefix_mine}Td_attempted"],
        "ctrl_sec": df[f"{prefix_mine}ctrl_sec"],
        "height_cm": df[f"{prefix_mine}Height_cm"],
        "reach_cm": df[f"{prefix_mine}Reach_cm"],
        "kd": df[f"{prefix_mine}KD"],
    })
    return out


long_df = pd.concat(
    [build_perspective(df, "f1", "f2"), build_perspective(df, "f2", "f1")],
    ignore_index=True,
)
long_df["sig_accuracy"] = np.where(
    long_df["sig_attempted"] > 0, long_df["sig_landed"] / long_df["sig_attempted"] * 100, np.nan
)
long_df["td_accuracy"] = np.where(
    long_df["td_attempted"] > 0, long_df["td_landed"] / long_df["td_attempted"] * 100, np.nan
)
long_df["result"] = np.where(long_df["won"] == 1, "Winner", "Loser")

# Restricted to fights with a real winner, for anything result-dependent
long_df_valid = long_df[long_df["has_winner"]]

# Fighter-level aggregate table for the search/comparison tool
fighter_stats = (
    long_df_valid.groupby("fighter")
    .agg(
        fights=("won", "size"),
        wins=("won", "sum"),
        sig_landed_sum=("sig_landed", "sum"),
        sig_attempted_sum=("sig_attempted", "sum"),
        td_landed_sum=("td_landed", "sum"),
        td_attempted_sum=("td_attempted", "sum"),
        avg_ctrl_sec=("ctrl_sec", "mean"),
        avg_sig_landed=("sig_landed", "mean"),
        height_cm=("height_cm", "max"),
        reach_cm=("reach_cm", "max"),
        stance=("stance", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
    )
    .reset_index()
)
fighter_stats["win_rate"] = (fighter_stats["wins"] / fighter_stats["fights"] * 100).round(1)
fighter_stats["sig_accuracy"] = np.where(
    fighter_stats["sig_attempted_sum"] > 0,
    fighter_stats["sig_landed_sum"] / fighter_stats["sig_attempted_sum"] * 100, np.nan,
).round(1)
fighter_stats["td_accuracy"] = np.where(
    fighter_stats["td_attempted_sum"] > 0,
    fighter_stats["td_landed_sum"] / fighter_stats["td_attempted_sum"] * 100, np.nan,
).round(1)
fighter_stats = fighter_stats[fighter_stats["fights"] >= 3].reset_index(drop=True)

FIGHTER_OPTIONS = sorted(fighter_stats["fighter"].dropna().unique().tolist())

# Control-time buckets for the win-rate-by-control panel
CTRL_BINS = [-0.01, 0, 30, 90, 300, 100000]
CTRL_LABELS = ["0s", "1-30s", "30-90s", "90s-5min", "5min+"]

# --------------------------------------------------------------------------
# 2. UFC-inspired color palette
# --------------------------------------------------------------------------

UFC_BLACK = "#0a0a0a"
UFC_PANEL = "#161616"
UFC_PANEL_BORDER = "#2a2a2a"
UFC_RED = "#d20a0a"
UFC_RED_BRIGHT = "#ff1f1f"
UFC_GOLD = "#e8b923"
UFC_TEXT = "#f2f2f2"
UFC_TEXT_MUTED = "#9a9a9a"
UFC_BLUE = "#5b8fd6"

METHOD_COLORS = {
    "Decision": "#9a9a9a",
    "KO/TKO": UFC_RED_BRIGHT,
    "Submission": UFC_GOLD,
    "DQ": UFC_BLUE,
    "Other": "#5c5c5c",
    "Overturned/No Winner": "#3a3a3a",
}
RESULT_COLORS = {"Winner": UFC_GOLD, "Loser": "#5c5c5c"}

PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor=UFC_PANEL,
    plot_bgcolor=UFC_PANEL,
    font=dict(color=UFC_TEXT, family="Inter, sans-serif"),
    xaxis=dict(gridcolor=UFC_PANEL_BORDER, zerolinecolor=UFC_PANEL_BORDER),
    yaxis=dict(gridcolor=UFC_PANEL_BORDER, zerolinecolor=UFC_PANEL_BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=30, l=10, r=10, b=10),
)

# --------------------------------------------------------------------------
# 3. App layout
# --------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "UFC Fight Analytics"
server = app.server

header = html.Div(
    className="header-bar",
    children=[
        html.Div(
            className="header-inner",
            children=[
                html.Span("UFC", className="header-logo"),
                html.Div([
                    html.H1("FIGHT ANALYTICS", className="header-title"),
                    html.P(
                        "Exploring win methods, striking, control time, stance, and "
                        "fighter matchups across 30+ years of UFC fights (1993-2025)",
                        className="header-subtitle",
                    ),
                ]),
            ],
        )
    ],
)

filters = dbc.Card(
    className="ufc-card filter-card",
    children=dbc.CardBody(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Year Range", className="filter-label"),
                        dcc.RangeSlider(
                            id="year-slider",
                            min=YEAR_MIN, max=YEAR_MAX,
                            value=[YEAR_MIN, YEAR_MAX],
                            step=1,
                            marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 5)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    md=8,
                ),
                dbc.Col(
                    [
                        html.Label("Weight Class", className="filter-label"),
                        dcc.Dropdown(
                            id="weightclass-dropdown",
                            options=[{"label": "All Weight Classes", "value": "ALL"}]
                            + [{"label": wc, "value": wc} for wc in WEIGHT_CLASSES],
                            value="ALL",
                            clearable=False,
                        ),
                    ],
                    md=4,
                ),
            ],
        )
    ),
)

kpi_row = dbc.Row(id="kpi-row", className="kpi-row g-3")

chart1_card = dbc.Card(
    className="ufc-card",
    children=[
        dbc.CardHeader("Fights Per Year, By Win Method"),
        dbc.CardBody(dcc.Graph(id="fights-per-year", config={"displayModeBar": False})),
    ],
)

chart2_card = dbc.Card(
    className="ufc-card",
    children=[
        dbc.CardHeader("Significant Strike Accuracy By Weight Class"),
        html.P(
            "Winners (gold) vs. losers (gray) - each point is one fighter's "
            "accuracy in one fight.",
            className="chart-note",
        ),
        dbc.CardBody(dcc.Graph(id="strike-accuracy", config={"displayModeBar": False})),
    ],
)

chart3_card = dbc.Card(
    className="ufc-card",
    children=[
        dbc.CardHeader("Control Time vs. Win Rate"),
        html.P(
            "Win rate for fighters grouped by how much control time they "
            "personally logged in the fight.",
            className="chart-note",
        ),
        dbc.CardBody(dcc.Graph(id="control-vs-winrate", config={"displayModeBar": False})),
    ],
)

stance_toggle = dbc.RadioItems(
    id="stance-view-mode",
    options=[
        {"label": "Overall", "value": "overall"},
        {"label": "By Weight Class", "value": "by_weight"},
    ],
    value="overall",
    inline=True,
    className="btn-group",
    inputClassName="btn-check",
    labelClassName="btn btn-outline-danger btn-sm",
    labelCheckedClassName="active",
)

chart4_card = dbc.Card(
    className="ufc-card",
    children=[
        dbc.CardHeader(
            dbc.Row(
                [
                    dbc.Col("Win Rate By Stance", width="auto"),
                    dbc.Col(stance_toggle, width="auto", className="ms-auto"),
                ],
                align="center", justify="between",
            )
        ),
        dbc.CardBody([
            html.Div(
                id="stance-back-wrapper",
                style={"display": "none"},
                children=[
                    dbc.Button(
                        "\u2190 All Weight Classes", id="stance-back-btn",
                        size="sm", color="danger", outline=True, className="mb-2",
                    ),
                ],
            ),
            html.Div(id="stance-breadcrumb", className="chart-note"),
            dcc.Store(id="stance-drill-store", data=None),
            dcc.Graph(id="stance-graph", config={"displayModeBar": False}),
        ]),
    ],
)

fighter_search_card = dbc.Card(
    className="ufc-card",
    children=[
        dbc.CardHeader("Fighter Search & Comparison"),
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Fighter A", className="filter-label"),
                            dcc.Dropdown(
                                id="fighter-a-dropdown",
                                options=[{"label": f, "value": f} for f in FIGHTER_OPTIONS],
                                value="Conor McGregor" if "Conor McGregor" in FIGHTER_OPTIONS else FIGHTER_OPTIONS[0],
                                clearable=False,
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Fighter B", className="filter-label"),
                            dcc.Dropdown(
                                id="fighter-b-dropdown",
                                options=[{"label": f, "value": f} for f in FIGHTER_OPTIONS],
                                value="Khabib Nurmagomedov" if "Khabib Nurmagomedov" in FIGHTER_OPTIONS else FIGHTER_OPTIONS[1],
                                clearable=False,
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            dcc.Graph(id="fighter-radar", config={"displayModeBar": False}),
            html.Div(id="fighter-compare-table"),
        ]),
    ],
)

app.layout = html.Div(
    style={"backgroundColor": UFC_BLACK, "minHeight": "100vh", "paddingBottom": "40px"},
    children=[
        header,
        dbc.Container(
            fluid=True,
            children=[
                filters,
                kpi_row,
                chart1_card,
                chart2_card,
                dbc.Row(
                    [
                        dbc.Col(chart3_card, md=6),
                        dbc.Col(chart4_card, md=6),
                    ]
                ),
                fighter_search_card,
            ],
        ),
    ],
)


# --------------------------------------------------------------------------
# 4. Helpers
# --------------------------------------------------------------------------

def filter_df(source, year_range, weight_class):
    lo, hi = year_range
    out = source[(source["year"] >= lo) & (source["year"] <= hi)]
    if weight_class != "ALL":
        out = out[out["weight_class"] == weight_class]
    return out


def kpi_card(value, label):
    return dbc.Col(
        html.Div(
            className="kpi-card",
            children=[
                html.Div(value, className="kpi-value"),
                html.Div(label, className="kpi-label"),
            ],
        ),
        md=3, sm=6, xs=12,
    )


# --------------------------------------------------------------------------
# 5. Callbacks
# --------------------------------------------------------------------------

@app.callback(
    Output("kpi-row", "children"),
    Input("year-slider", "value"),
    Input("weightclass-dropdown", "value"),
)
def update_kpis(year_range, weight_class):
    dff = filter_df(df, year_range, weight_class)
    total_fights = len(dff)
    total_events = dff["event_name"].nunique()
    top_method = dff.loc[dff["has_winner"], "method_group"].mode()
    top_method = top_method.iloc[0] if not top_method.empty else "N/A"
    lo, hi = year_range

    return [
        kpi_card(f"{total_fights:,}", "Total Fights"),
        kpi_card(f"{total_events:,}", "Total Events"),
        kpi_card(f"{lo}-{hi}", "Year Range"),
        kpi_card(top_method, "Most Common Finish"),
    ]


@app.callback(
    Output("fights-per-year", "figure"),
    Input("year-slider", "value"),
    Input("weightclass-dropdown", "value"),
)
def update_fights_per_year(year_range, weight_class):
    dff = filter_df(df, year_range, weight_class)
    counts = dff.groupby(["year", "method_group"]).size().reset_index(name="count")

    fig = px.bar(
        counts, x="year", y="count", color="method_group",
        category_orders={"method_group": METHOD_ORDER},
        color_discrete_map=METHOD_COLORS,
        labels={"year": "Year", "count": "Number of Fights", "method_group": "Method"},
    )
    fig.update_layout(barmode="stack", legend_title_text="Win Method", **PLOTLY_DARK_LAYOUT)
    fig.update_xaxes(dtick=2)
    return fig


@app.callback(
    Output("strike-accuracy", "figure"),
    Input("year-slider", "value"),
    Input("weightclass-dropdown", "value"),
)
def update_strike_accuracy(year_range, weight_class):
    lo, hi = year_range
    dff = long_df_valid[(long_df_valid["year"] >= lo) & (long_df_valid["year"] <= hi)]
    if weight_class != "ALL":
        dff = dff[dff["weight_class"] == weight_class]
    dff = dff[dff["sig_attempted"] >= 5]  # drop near-zero-attempt noise

    order = [w for w in WEIGHT_CLASSES if w in dff["weight_class"].unique()]
    fig = px.box(
        dff, x="weight_class", y="sig_accuracy", color="result",
        category_orders={"weight_class": order, "result": ["Winner", "Loser"]},
        color_discrete_map=RESULT_COLORS,
        points=False,
        labels={"weight_class": "Weight Class", "sig_accuracy": "Sig. Strike Accuracy (%)", "result": ""},
    )
    fig.update_layout(boxmode="group", legend_title_text="", **PLOTLY_DARK_LAYOUT)
    fig.update_xaxes(tickangle=-30)
    return fig


@app.callback(
    Output("control-vs-winrate", "figure"),
    Input("year-slider", "value"),
    Input("weightclass-dropdown", "value"),
)
def update_control_vs_winrate(year_range, weight_class):
    lo, hi = year_range
    bucket_df = long_df_valid[(long_df_valid["year"] >= lo) & (long_df_valid["year"] <= hi)]
    if weight_class != "ALL":
        bucket_df = bucket_df[bucket_df["weight_class"] == weight_class]
    bucket_df = bucket_df[bucket_df["ctrl_sec"].notna()].copy()
    bucket_df["ctrl_bucket"] = pd.cut(bucket_df["ctrl_sec"], bins=CTRL_BINS, labels=CTRL_LABELS)
    summary = (
        bucket_df.groupby("ctrl_bucket", observed=True)
        .agg(win_rate=("won", "mean"), n=("won", "size"))
        .reindex(CTRL_LABELS)
        .reset_index()
    )
    summary["win_rate"] = summary["win_rate"] * 100

    fig = go.Figure()
    fig.add_bar(
        x=summary["ctrl_bucket"], y=summary["win_rate"],
        marker_color=UFC_RED_BRIGHT,
        text=[f"{v:.0f}%<br>(n={n:,})" if not pd.isna(v) else "" for v, n in zip(summary["win_rate"], summary["n"])],
        textposition="outside",
    )
    fig.add_hline(y=50, line_dash="dash", line_color=UFC_TEXT_MUTED,
                  annotation_text="50% (coin flip)", annotation_position="bottom right",
                  annotation_font_color=UFC_TEXT_MUTED)
    layout = dict(PLOTLY_DARK_LAYOUT)
    layout.update(
        yaxis_title="Win Rate (%)", xaxis_title="Own Control Time",
        yaxis_range=[0, 100],
    )
    fig.update_layout(**layout)
    return fig


@app.callback(
    Output("stance-drill-store", "data"),
    Input("stance-view-mode", "value"),
    Input("stance-graph", "clickData"),
    Input("stance-back-btn", "n_clicks"),
    State("stance-drill-store", "data"),
)
def manage_stance_drill(view_mode, click_data, back_clicks, current):
    trigger = ctx.triggered_id
    if trigger == "stance-view-mode":
        return None  # reset drill-down whenever the toggle changes
    if trigger == "stance-back-btn":
        return None
    if trigger == "stance-graph" and view_mode == "by_weight" and current is None and click_data:
        return click_data["points"][0]["x"]
    return current


@app.callback(
    Output("stance-graph", "figure"),
    Output("stance-back-wrapper", "style"),
    Output("stance-breadcrumb", "children"),
    Input("year-slider", "value"),
    Input("weightclass-dropdown", "value"),
    Input("stance-view-mode", "value"),
    Input("stance-drill-store", "data"),
)
def update_stance_chart(year_range, weight_class, view_mode, drill_selected):
    lo, hi = year_range
    dff = long_df_valid[(long_df_valid["year"] >= lo) & (long_df_valid["year"] <= hi)]
    dff = dff[dff["stance"].isin(MAIN_STANCES)]
    if weight_class != "ALL":
        dff = dff[dff["weight_class"] == weight_class]

    def stance_bar(sub_df, title_note):
        summary = sub_df.groupby("stance").agg(win_rate=("won", "mean"), n=("won", "size")).reset_index()
        summary["win_rate"] = (summary["win_rate"] * 100).round(1)
        summary = summary.sort_values("win_rate", ascending=False)
        fig = go.Figure()
        fig.add_bar(
            x=summary["stance"], y=summary["win_rate"],
            text=[f"{wr}%<br>({n:,} fights)" for wr, n in zip(summary["win_rate"], summary["n"])],
            textposition="outside", marker_color=UFC_RED_BRIGHT,
        )
        fig.add_hline(y=50, line_dash="dash", line_color=UFC_TEXT_MUTED,
                       annotation_text="50% (coin flip)", annotation_position="bottom right",
                       annotation_font_color=UFC_TEXT_MUTED)
        layout = dict(PLOTLY_DARK_LAYOUT)
        layout.update(yaxis_title="Win Rate (%)", xaxis_title="Stance",
                       yaxis_range=[0, max(60, (summary["win_rate"].max() if len(summary) else 0) + 10)])
        fig.update_layout(**layout)
        return fig

    if view_mode == "overall":
        return stance_bar(dff, ""), {"display": "none"}, ""

    # --- by_weight mode: drill-down ---
    if drill_selected is None:
        # Level 1: aggregate win rate per weight class (click a bar to drill in)
        order = [w for w in WEIGHT_CLASSES if w in dff["weight_class"].unique()]
        summary = dff.groupby("weight_class").agg(win_rate=("won", "mean"), n=("won", "size")).reindex(order).reset_index()
        summary["win_rate"] = (summary["win_rate"] * 100).round(1)

        fig = go.Figure()
        fig.add_bar(
            x=summary["weight_class"], y=summary["win_rate"],
            text=[f"{wr:.0f}%<br>(n={n:,.0f})" if pd.notna(wr) else "" for wr, n in zip(summary["win_rate"], summary["n"])],
            textposition="outside", marker_color=UFC_GOLD,
        )
        fig.add_hline(y=50, line_dash="dash", line_color=UFC_TEXT_MUTED)
        layout = dict(PLOTLY_DARK_LAYOUT)
        layout.update(yaxis_title="Win Rate (%)", xaxis_title="",
                       yaxis_range=[0, max(60, (summary["win_rate"].max() if len(summary) else 0) + 15)])
        fig.update_layout(**layout)
        fig.update_xaxes(tickangle=-30)
        breadcrumb = "All weight classes \u2014 click a bar to see its stance breakdown"
        return fig, {"display": "none"}, breadcrumb

    # Level 2: stance breakdown for the selected weight class
    sub = dff[dff["weight_class"] == drill_selected]
    fig = stance_bar(sub, drill_selected)
    breadcrumb = f"All Weight Classes  /  {drill_selected}"
    return fig, {"display": "block"}, breadcrumb


@app.callback(
    Output("fighter-radar", "figure"),
    Output("fighter-compare-table", "children"),
    Input("fighter-a-dropdown", "value"),
    Input("fighter-b-dropdown", "value"),
)
def update_fighter_comparison(fighter_a, fighter_b):
    row_a = fighter_stats[fighter_stats["fighter"] == fighter_a]
    row_b = fighter_stats[fighter_stats["fighter"] == fighter_b]

    metrics = ["win_rate", "sig_accuracy", "td_accuracy"]
    metric_labels = ["Win Rate %", "Sig. Strike Acc. %", "Takedown Acc. %"]
    # normalize avg control time & avg sig strikes landed to 0-100 vs population max for radar comparability
    ctrl_max = fighter_stats["avg_ctrl_sec"].max()
    landed_max = fighter_stats["avg_sig_landed"].max()

    def build_values(row):
        if row.empty:
            return [0] * 5
        r = row.iloc[0]
        return [
            r["win_rate"],
            r["sig_accuracy"] if pd.notna(r["sig_accuracy"]) else 0,
            r["td_accuracy"] if pd.notna(r["td_accuracy"]) else 0,
            (r["avg_ctrl_sec"] / ctrl_max * 100) if pd.notna(r["avg_ctrl_sec"]) else 0,
            (r["avg_sig_landed"] / landed_max * 100) if pd.notna(r["avg_sig_landed"]) else 0,
        ]

    categories = metric_labels + ["Avg Control Time (rel.)", "Avg Strikes Landed (rel.)"]
    values_a = build_values(row_a)
    values_b = build_values(row_b)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_a + values_a[:1], theta=categories + categories[:1],
        fill="toself", name=fighter_a, line_color=UFC_RED_BRIGHT,
        fillcolor="rgba(255,31,31,0.25)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=values_b + values_b[:1], theta=categories + categories[:1],
        fill="toself", name=fighter_b, line_color=UFC_GOLD,
        fillcolor="rgba(232,185,35,0.20)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=UFC_PANEL,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=UFC_PANEL_BORDER, color=UFC_TEXT_MUTED),
            angularaxis=dict(gridcolor=UFC_PANEL_BORDER, color=UFC_TEXT),
        ),
        showlegend=True,
        **{k: v for k, v in PLOTLY_DARK_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )

    def stat_row(label, a_val, b_val):
        return html.Tr([html.Td(label, className="compare-label"),
                         html.Td(a_val, className="compare-value"),
                         html.Td(b_val, className="compare-value")])

    def fmt(row, col, suffix=""):
        if row.empty or pd.isna(row.iloc[0][col]):
            return "N/A"
        val = row.iloc[0][col]
        return f"{val:.0f}{suffix}" if isinstance(val, (int, float, np.floating)) else str(val)

    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th(""), html.Th(fighter_a), html.Th(fighter_b)])),
            html.Tbody([
                stat_row("Fights", fmt(row_a, "fights"), fmt(row_b, "fights")),
                stat_row("Wins", fmt(row_a, "wins"), fmt(row_b, "wins")),
                stat_row("Win Rate", fmt(row_a, "win_rate", "%"), fmt(row_b, "win_rate", "%")),
                stat_row("Sig. Strike Accuracy", fmt(row_a, "sig_accuracy", "%"), fmt(row_b, "sig_accuracy", "%")),
                stat_row("Takedown Accuracy", fmt(row_a, "td_accuracy", "%"), fmt(row_b, "td_accuracy", "%")),
                stat_row("Height (cm)", fmt(row_a, "height_cm"), fmt(row_b, "height_cm")),
                stat_row("Reach (cm)", fmt(row_a, "reach_cm"), fmt(row_b, "reach_cm")),
                stat_row("Stance", fmt(row_a, "stance"), fmt(row_b, "stance")),
            ]),
        ],
        bordered=False, className="compare-table", size="sm",
    )
    return fig, table


# --------------------------------------------------------------------------
# 6. Run
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
