"""
UFC Fights Dashboard (1993-2025)
--------------------------------
An interactive Plotly Dash app exploring:
  1. Fights per year, broken down by win method
  2. Control time vs. win method
  3. Win rate by fighter stance

Layout inspired by common "history dashboard" templates (dark theme, header
banner, KPI stat cards, chart cards), restyled with a UFC-inspired black /
red / gold palette. No UFC trademarks or logos are used - only the color
language.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:8050 in your browser.

Data assumption: following the UFCStats.com convention that this dataset
was scraped from, `fighter_1` is the winner of the bout and `fighter_2`
is the loser, except for rows where `method` is "Overturned" or
"Could Not Continue" (no meaningful winner) - those are excluded from
any analysis that depends on knowing who won, but still counted in the
raw "fights per year" totals.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output

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
WEIGHT_CLASSES = sorted(df["weight_class"].dropna().unique().tolist())

_valid = df[df["has_winner"]]
_as_f1 = _valid[["f1_Stance", "weight_class", "year"]].rename(columns={"f1_Stance": "stance"})
_as_f1["won"] = 1
_as_f2 = _valid[["f2_Stance", "weight_class", "year"]].rename(columns={"f2_Stance": "stance"})
_as_f2["won"] = 0
fighter_long = pd.concat([_as_f1, _as_f2], ignore_index=True)
MAIN_STANCES = ["Orthodox", "Southpaw", "Switch"]
fighter_long = fighter_long[fighter_long["stance"].isin(MAIN_STANCES)]

METHOD_ORDER = ["Decision", "KO/TKO", "Submission", "DQ", "Other", "Overturned/No Winner"]

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

METHOD_COLORS = {
    "Decision": "#9a9a9a",        # neutral gray - default outcome
    "KO/TKO": UFC_RED_BRIGHT,     # red - the violent finish
    "Submission": UFC_GOLD,       # gold - the technical finish
    "DQ": "#5b8fd6",
    "Other": "#5c5c5c",
    "Overturned/No Winner": "#3a3a3a",
}

PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor=UFC_PANEL,
    plot_bgcolor=UFC_PANEL,
    font=dict(color=UFC_TEXT, family="Inter, sans-serif"),
    xaxis=dict(gridcolor=UFC_PANEL_BORDER, zerolinecolor=UFC_PANEL_BORDER),
    yaxis=dict(gridcolor=UFC_PANEL_BORDER, zerolinecolor=UFC_PANEL_BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=20, l=10, r=10, b=10),
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
                        "Exploring win methods, control time, and stance across "
                        "30+ years of UFC fights (1993-2025)",
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
        dbc.CardHeader("Control Time vs. Win Method"),
        dbc.CardBody(dcc.Graph(id="control-vs-method", config={"displayModeBar": False})),
    ],
)

chart3_card = dbc.Card(
    className="ufc-card",
    children=[
        dbc.CardHeader("Win Rate By Stance"),
        dbc.CardBody(dcc.Graph(id="stance-winrate", config={"displayModeBar": False})),
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
                dbc.Row(
                    [
                        dbc.Col(chart2_card, md=6),
                        dbc.Col(chart3_card, md=6),
                    ]
                ),
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
    total_events = dff["event_name"].nunique() if "event_name" in dff.columns else dff["event_date"].nunique()
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
    Output("control-vs-method", "figure"),
    Input("year-slider", "value"),
    Input("weightclass-dropdown", "value"),
)
def update_control_vs_method(year_range, weight_class):
    dff = filter_df(df, year_range, weight_class)
    dff = dff[dff["has_winner"] & dff["winner_ctrl_sec"].notna()]
    dff = dff[dff["method_group"] != "Overturned/No Winner"]

    present_order = [m for m in METHOD_ORDER if m in dff["method_group"].unique()]
    fig = px.box(
        dff, x="method_group", y="winner_ctrl_sec", color="method_group",
        category_orders={"method_group": present_order},
        color_discrete_map=METHOD_COLORS,
        points=False,
        labels={"method_group": "Win Method", "winner_ctrl_sec": "Winner's Control Time (s)"},
    )
    fig.update_layout(showlegend=False, **PLOTLY_DARK_LAYOUT)
    return fig


@app.callback(
    Output("stance-winrate", "figure"),
    Input("year-slider", "value"),
    Input("weightclass-dropdown", "value"),
)
def update_stance_winrate(year_range, weight_class):
    lo, hi = year_range
    dff = fighter_long[(fighter_long["year"] >= lo) & (fighter_long["year"] <= hi)]
    if weight_class != "ALL":
        dff = dff[dff["weight_class"] == weight_class]

    summary = (
        dff.groupby("stance").agg(win_rate=("won", "mean"), fights=("won", "size")).reset_index()
    )
    summary["win_rate"] = (summary["win_rate"] * 100).round(1)
    summary = summary.sort_values("win_rate", ascending=False)

    fig = go.Figure()
    fig.add_bar(
        x=summary["stance"], y=summary["win_rate"],
        text=[f"{wr}%<br>({n:,} fights)" for wr, n in zip(summary["win_rate"], summary["fights"])],
        textposition="outside",
        marker_color=UFC_RED_BRIGHT,
    )
    fig.add_hline(
        y=50, line_dash="dash", line_color=UFC_TEXT_MUTED,
        annotation_text="50% (coin flip)", annotation_position="bottom right",
        annotation_font_color=UFC_TEXT_MUTED,
    )
    layout = dict(PLOTLY_DARK_LAYOUT)
    layout.update(
        yaxis_title="Win Rate (%)", xaxis_title="Stance",
        yaxis_range=[0, max(60, summary["win_rate"].max() + 10)],
    )
    fig.update_layout(**layout)
    return fig


# --------------------------------------------------------------------------
# 6. Run
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
