from dash import Dash, dcc, html, Input, Output, State
import dash
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
import pandas as pd
import requests
import io
# data ------------------------------------------------------------------data = np.load("data/chr1/demo_pca_chr1.npy", allow_pickle=True).item()

DATA_URL = (
  "https://raw.githubusercontent.com/cs237-final/src/main/"
  "data/chr1/demo_pca_chr1.npy"
)
data = np.load(io.BytesIO(requests.get(DATA_URL).content), allow_pickle=True).item()

with open("data/chr1/hover-interp.json", "r") as f:
   hover_texts = json.load(f)




pca = data["pca"]
pcs = data["pcs"]
superpops = data["superpops"]
samples = data.get("samples", np.arange(pcs.shape[0]))

# colors and labels ------------------------------------------------------
superpop2color = {
    "AFR": "teal",
    "AMR": "orange",
    "EAS": "olive",
    "EUR": "magenta",
    "SAS": "purple",
}

code2label = {
    "AFR": "Africa",
    "EUR": "Europe",
    "EAS": "East Asia",
    "SAS": "South Asia",
    "AMR": "Americas",
}

# hover label function ---------------------------------------------------
def make_label(hover_text_json, num_pcs=3, num_snps=3, desc=False):
    texts = []
    if not desc:
        for sample in hover_text_json.keys():
            pc_nums = hover_text_json[sample]["top_pcs"][:num_pcs]

            # sample line
            text = f"<b>Sample: {sample}</b><br>"

            for i in range(num_pcs):
                pc_data = hover_text_json[sample][str(1 + i)]
                pc = pc_data["pc"]
                text += f"<b>PC {pc}</b><br>"

                top_feats = pc_data["features"][:num_snps]
                genes = [description[3].split()[0] for description in top_feats]

                for j, feat in enumerate(top_feats):
                    pos = feat[1]
                    text += f"{genes[j]}, Position: {pos} (PC {pc})<br>"

            texts.append(text)
    return texts

# dataframe for plot -----------------------------------------------------
def build_df_plot(num_pcs, num_snps):
    hovers = make_label(hover_texts, num_pcs=num_pcs, num_snps=num_snps)
    df_plot = pd.DataFrame(
        {
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "PC3": pcs[:, 2],
            "PC4": pcs[:, 3],
            "PC5": pcs[:, 4],
            "sample": samples,
            "hover": hovers,
            "Superpopulation": superpops,
        }
    )
    return df_plot

# figure builder ---------------------------------------------------------
def build_figure(num_pcs, num_snps):
    df_plot = build_df_plot(num_pcs, num_snps)
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        vertical_spacing=0.05,
        subplot_titles=["PC1 vs PC2", "PC2 vs PC3", "PC3 vs PC4", "PC4 vs PC5"],
        row_heights=[0.7, 0.3],
        column_widths=[0.5, 0.5, 0.5],
    )

    # main
    fig.add_trace(
        go.Scatter(
            x=df_plot["PC1"],
            y=df_plot["PC2"],
            mode="markers",
            marker=dict(
                color=df_plot["Superpopulation"].map(superpop2color),
                size=5,
                opacity=0.7,
                line=dict(width=0),
            ),
            text=df_plot["hover"],
            hovertemplate="%{text}",
            name="PC1 vs PC2",
        ),
        row=1,
        col=1,
    )

    # subplots
    fig.add_trace(
        go.Scatter(
            x=df_plot["PC2"],
            y=df_plot["PC3"],
            mode="markers",
            marker=dict(
                color=df_plot["Superpopulation"].map(superpop2color),
                size=5,
                opacity=0.7,
                line=dict(width=0),
            ),
            text=df_plot["hover"],
            hovertemplate="%{text}",
            name="PC2 vs PC3",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot["PC3"],
            y=df_plot["PC4"],
            mode="markers",
            marker=dict(
                color=df_plot["Superpopulation"].map(superpop2color),
                size=5,
                opacity=0.7,
                line=dict(width=0),
            ),
            text=df_plot["hover"],
            hovertemplate="%{text}",
            name="PC3 vs PC4",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot["PC4"],
            y=df_plot["PC5"],
            mode="markers",
            marker=dict(
                color=df_plot["Superpopulation"].map(superpop2color),
                size=5,
                opacity=0.7,
                line=dict(width=0),
            ),
            text=df_plot["hover"],
            hovertemplate="%{text}",
            name="PC4 vs PC5",
        ),
        row=2,
        col=3,
    )

    fig.update_traces(showlegend=False)

    # legend
    for sp, color in superpop2color.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=color, size=8),
                name=code2label[sp],
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        height=800,
        width=1000,
        legend_title_text="Superpopulation",
        clickmode="event+select",
        hovermode="closest",
    )
    return fig, df_plot

# app --------------------------------------------------------------------
app = Dash("yass")

# controls box -----------------------------------------------------------
controls_box = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Number of PCs (1–5)", style={"marginRight": "6px"}),
                        dcc.Input(
                            id="num-pcs",
                            type="number",
                            min=1,
                            max=5,
                            step=1,
                            value=3,
                            style={
                                "width": "70px",
                                "borderRadius": "16px",
                                "border": "1px solid #ccc",
                                "padding": "2px 6px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginRight": "20px",
                    },
                ),
                html.Div(
                    [
                        html.Label("Number of SNPs (1–10)", style={"marginRight": "6px"}),
                        dcc.Input(
                            id="num-snps",
                            type="number",
                            min=1,
                            max=10,
                            step=1,
                            value=3,
                            style={
                                "width": "70px",
                                "borderRadius": "16px",
                                "border": "1px solid #ccc",
                                "padding": "2px 6px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginTop": "10px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "left",
                "justifyContent": "center",
                "flexWrap": "wrap",
            },
        )
    ],
    style={
        "backgroundColor": "#e5ecf6",
        "borderRadius": "8px",
        "padding": "8px 14px",
        "marginTop": "10px",
        "marginBottom": "10px",
        "border": "none",
        "maxWidth": "380px",
        "width": "100%",
        "marginLeft": "auto",
        "marginRight": "auto",
    },
)

# aggregate box ----------------------------------------------------------
aggregate_box = html.Div(
    [
        html.Button(
            "Aggregate data from lasso selection",
            id="aggregate-button",
            n_clicks=0,
            className="hover-button",
            style={
                "marginBottom": "6px",
                "padding": "4px 8px",
                "fontSize": "12px",
            },
        ),
        html.Button(
            "Clear lasso",
            id="clear-lasso-button",
            n_clicks=0,
            className="hover-button",
            style={
                "padding": "4px 8px",
                "fontSize": "12px",
            },
        ),
    ],
    style={
        "backgroundColor": "#e5ecf6",
        "borderRadius": "8px",
        "padding": "8px 14px",
        "marginTop": "10px",
        "marginBottom": "10px",
        "border": "none",
        "maxWidth": "260px",
        "width": "100%",
    },
)

# right panel ------------------------------------------------------------
right_panel = html.Div(
    style={
        "flex": "2",
        "backgroundColor": "#e5ecf6",
        "borderRadius": "8px",
        "padding": "12px 14px",
        "border": "none",
        "overflowY": "auto",
        "maxHeight": "800px",
    },
    children=[
        html.H3(
            "Interpretation Panel",
            style={
                "marginTop": "10px",
                "marginBottom": "10px",
                "fontSize": "20px",
                "fontWeight": "bold",
            },
        ),
        html.Div(
            [
                dcc.Input(
                    id="sample-search",
                    type="text",
                    placeholder="Search by sample ID (ex. NA19711)",
                    style={
                        "width": "95%",
                        "marginBottom": "10px",
                        "padding": "6px 10px",
                        "borderRadius": "16px",
                        "border": "1px solid #ccc",
                        "outline": "none",
                    },
                )
            ],
            style={"marginBottom": "8px"},
        ),
        html.Div(
            id="interpretation-wrapper",
            style={
                "backgroundColor": "white",
                "borderRadius": "8px",
                "padding": "8px 10px",
                "boxShadow": "0 0 4px rgba(0,0,0,0.05)",
            },
            children=[
                html.Div(
                    id="interpretation-message",
                    style={"marginBottom": "8px", "fontSize": "12px"},
                ),
                html.Table(
                    id="interpretation-panel",
                    style={
                        "width": "100%",
                        "borderCollapse": "collapse",
                        "fontSize": "13px",
                    },
                ),
            ],
        ),
    ],
)

# layout -----------------------------------------------------------------
app.layout = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "gap": "20px",
        "fontFamily": "Helvetica, Arial, sans-serif",
        "padding": "16px",
    },
    children=[
        html.H1(
            "PCA Interpretations",
            style={
                "margin": 0,
                "fontSize": "28px",
                "fontWeight": "bold",
                "textAlign": "center",
            },
        ),
        html.Div(
            style={"display": "flex", "flexDirection": "row", "gap": "20px"},
            children=[
                html.Div(
                    style={
                        "flex": "3",
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                    },
                    children=[
                        html.Div(
                            [controls_box, aggregate_box],
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "justifyContent": "space-between",
                                "alignItems": "flex-start",
                                "width": "100%",
                                "maxWidth": "800px",
                            },
                        ),
                        dcc.Graph(
                            id="pca-graph",
                            clear_on_unhover=True,
                            style={"height": "800px", "width": "100%"},
                        ),
                        dcc.Store(id="df-plot-store"),
                        dcc.Store(id="orig-marker-store"),
                    ],
                ),
                right_panel,
            ],
        ),
    ],
)

# callbacks --------------------------------------------------------------
@app.callback(
    Output("pca-graph", "figure"),
    Output("df-plot-store", "data"),
    Output("orig-marker-store", "data"),
    Input("num-pcs", "value"),
    Input("num-snps", "value"),
)
def update_figure(num_pcs, num_snps):
    if num_pcs is None:
        num_pcs = 3
    if num_snps is None:
        num_snps = 3
    num_pcs = max(1, min(5, int(num_pcs)))
    num_snps = max(1, min(10, int(num_snps)))
    fig, df_plot = build_figure(num_pcs, num_snps)

    orig = {}
    for i_trace in [1, 2, 3]:
        if i_trace >= len(fig.data):
            continue
        t = fig.data[i_trace]
        if isinstance(t.marker.size, (int, float)):
            sizes = [t.marker.size] * len(t.marker.color)
        else:
            sizes = list(t.marker.size)
        orig[str(i_trace)] = {"color": list(t.marker.color), "size": sizes}

    return fig, df_plot.to_json(date_format="iso", orient="split"), orig

BASE_SIZE = 5
HOVER_SIZE = 15
HOVER_COLOR = "red"

@app.callback(
    Output("pca-graph", "figure", allow_duplicate=True),
    Input("pca-graph", "hoverData"),
    State("pca-graph", "figure"),
    State("orig-marker-store", "data"),
    prevent_initial_call=True,
)
def sync_hover(hoverData, fig, orig):
    if hoverData is None or orig is None:
        return fig
    raw_point = hoverData.get("points", [None])[0]
    point = raw_point[0] if isinstance(raw_point, list) and raw_point else raw_point
    if not isinstance(point, dict):
        return fig
    idx = point.get("pointIndex")
    if idx is None:
        return fig

    for i_trace in [1, 2, 3]:
        if i_trace >= len(fig["data"]):
            continue
        key = str(i_trace)
        if key not in orig:
            continue
        trace = fig["data"][i_trace]
        trace["marker"]["line"] = {"width": 0}
        base_colors = orig[key]["color"]
        base_sizes = orig[key]["size"]
        colors = list(base_colors)
        sizes = list(base_sizes)
        if 0 <= idx < len(colors):
            colors[idx] = HOVER_COLOR
            sizes[idx] = HOVER_SIZE
        trace["marker"]["color"] = colors
        trace["marker"]["size"] = sizes
    return fig

@app.callback(
    Output("pca-graph", "figure", allow_duplicate=True),
    Input("pca-graph", "relayoutData"),
    State("pca-graph", "figure"),
    State("orig-marker-store", "data"),
    prevent_initial_call=True,
)
def reset_hover(relayoutData, fig, orig):
    if relayoutData is None or orig is None:
        return fig
    for i_trace in [1, 2, 3]:
        if i_trace >= len(fig["data"]):
            continue
        key = str(i_trace)
        if key not in orig:
            continue
        trace = fig["data"][i_trace]
        trace["marker"]["line"] = {"width": 0}
        trace["marker"]["color"] = list(orig[key]["color"])
        trace["marker"]["size"] = list(orig[key]["size"])
    return fig

# per-sample table -------------------------------------------------------
def build_interpretation_table(idx, df_plot, num_pcs, num_snps):
    if num_pcs is None:
        num_pcs = 3
    if num_snps is None:
        num_snps = 3
    num_pcs = max(1, min(5, int(num_pcs)))
    num_snps = max(1, min(10, int(num_snps)))

    sample_name = df_plot.loc[idx, "sample"]
    sample_data = hover_texts.get(str(sample_name)) or hover_texts.get(sample_name)

    header = html.Thead(
        html.Tr(
            [
                html.Th("PC", style={"textAlign": "left", "padding": "4px"}),
                html.Th("Gene", style={"textAlign": "left", "padding": "4px"}),
                html.Th("Function", style={"textAlign": "left", "padding": "4px"}),
            ]
        )
    )

    if sample_data is None:
        body = html.Tbody(
            html.Tr(
                html.Td(
                    f"No interpretation found for sample {sample_name}.",
                    colSpan=3,
                    style={"padding": "4px"},
                )
            )
        )
        return header, body

    rows = []
    top_pcs_order = sample_data.get("top_pcs", [])[:num_pcs]
    for rank, pc_idx in enumerate(top_pcs_order, start=1):
        pc_key = str(rank)
        pc_block = sample_data.get(pc_key)
        if pc_block is None:
            continue
        pc_id = pc_block["pc"]
        features = pc_block["features"][:num_snps]
        for feat in features:
            pos = feat[1]
            full_desc = feat[3]
            if " (" in full_desc:
                gene, rest = full_desc.split(" (", 1)
                gene = gene.strip()
                func = rest.rsplit(")", 1)[0].strip()
            else:
                gene = full_desc.strip()
                func = ""
            rows.append(
                html.Tr(
                    [
                        html.Td(f"PC {pc_id}", style={"padding": "4px"}),
                        html.Td(f"{gene} (pos {pos})", style={"padding": "4px"}),
                        html.Td(func, style={"padding": "4px"}),
                    ]
                )
            )

    if not rows:
        body = html.Tbody(
            html.Tr(
                html.Td(
                    "No features found for this sample.",
                    colSpan=3,
                    style={"padding": "4px"},
                )
            )
        )
        return header, body

    body = html.Tbody(rows)
    return header, body

# click: message + table -------------------------------------------------
@app.callback(
    Output("interpretation-message", "children"),
    Output("interpretation-panel", "children"),
    Input("pca-graph", "clickData"),
    State("df-plot-store", "data"),
    State("num-pcs", "value"),
    State("num-snps", "value"),
)
def update_interpretation(clickData, df_json, num_pcs, num_snps):
    message = (
        "Click a point or search for a sample to explore differentiating features. "
        "To explore key genes in multiple samples, use the lasso tool."
    )

    if clickData is None or df_json is None:
        header = html.Thead(
            html.Tr(
                [
                    html.Th("PC", style={"textAlign": "left", "padding": "4px"}),
                    html.Th("Gene", style={"textAlign": "left", "padding": "4px"}),
                    html.Th("Function", style={"textAlign": "left", "padding": "4px"}),
                ]
            )
        )
        body = html.Tbody([])
        return message, [header, body]

    df_plot = pd.read_json(df_json, orient="split")
    raw_point = clickData.get("points", [None])[0]
    point = raw_point[0] if isinstance(raw_point, list) and raw_point else raw_point
    if not isinstance(point, dict):
        return dash.no_update, dash.no_update
    idx = point.get("pointIndex")
    if idx is None:
        return dash.no_update, dash.no_update

    header, body = build_interpretation_table(idx, df_plot, num_pcs, num_snps)
    return message, [header, body]
@app.callback(
    Output('interpretation-message', 'children', allow_duplicate=True),
    Output('interpretation-panel', 'children', allow_duplicate=True),
    Input('sample-search', 'value'),
    State('df-plot-store', 'data'),
    State('num-pcs', 'value'),
    State('num-snps', 'value'),
    prevent_initial_call=True,
)
def update_interpretation_by_sample(sample_name, dfjson, numpcs, numsnps):
    # If nothing typed or no data yet, do not change anything
    if not sample_name or dfjson is None:
        return dash.no_update, dash.no_update

    # Normalize sample name as string
    sample_name = str(sample_name).strip()
    if not sample_name:
        return dash.no_update, dash.no_update

    # Rebuild dfplot from store
    try:
        dfplot = pd.read_json(dfjson, orient='split')
    except Exception:
        return dash.no_update, dash.no_update

    # Try to find exact match in the 'sample' column
    if sample_name not in dfplot['sample'].values:
        # Optional: also allow case‑insensitive match
        matches = dfplot[dfplot['sample'].str.upper() == sample_name.upper()]
        if matches.empty:
            # No matching sample found – show message and empty table
            header = html.Thead(
                html.Tr([
                    html.Th("PC", style={"textAlign": "left", "padding": "4px"}),
                    html.Th("Gene", style={"textAlign": "left", "padding": "4px"}),
                    html.Th("Function", style={"textAlign": "left", "padding": "4px"}),
                ])
            )
            body = html.Tbody(
                html.Tr(
                    html.Td(
                        f"No sample found with ID '{sample_name}'.",
                        colSpan=3,
                        style={"padding": "4px"},
                    )
                )
            )
            msg = (
                "Search by exact sample ID present in the PCA data, "
                "or click a point on the plot."
            )
            return msg, [header, body]
        # Use the first matched index if case‑insensitive match succeeded
        idx = matches.index[0]
    else:
        # Exact match
        idx = dfplot.index[dfplot['sample'] == sample_name][0]

    # Reuse your existing table builder
    header, body = build_interpretation_table(idx, dfplot, numpcs, numsnps)

    message = (
        f"Showing interpretation for sample {sample_name}. "
        "You can also click a point or use lasso to compare multiple samples."
    )
    return message, [header, body]

@app.callback(
    Output("interpretation-message", "children", allow_duplicate=True),
    Output("interpretation-panel", "children", allow_duplicate=True),
    Input("aggregate-button", "n_clicks"),
    State("pca-graph", "selectedData"),
    State("df-plot-store", "data"),
    State("num-pcs", "value"),
    State("num-snps", "value"),
    prevent_initial_call=True,
)
def aggregate_lasso(n_clicks, selectedData, df_json, num_pcs, num_snps):
    if not n_clicks:
        return dash.no_update, dash.no_update

    if selectedData is None or df_json is None:
        return (
            "Lasso points on the main plot, then click 'Aggregate data from lasso selection'.",
            [],
        )

    df_plot = pd.read_json(df_json, orient="split")

    # collect indices from main trace (curveNumber 0)
    selected_indices = []
    for pt in selectedData.get("points", []):
        if pt.get("curveNumber") == 0:
            idx = pt.get("pointIndex")
            if idx is not None:
                selected_indices.append(idx)

    selected_indices = sorted(set(selected_indices))
    if not selected_indices:
        return ("No points selected in the main PCA plot.", [])

    if num_pcs is None:
        num_pcs = 3
    if num_snps is None:
        num_snps = 3
    num_pcs = max(1, min(5, int(num_pcs)))
    num_snps = max(1, min(10, int(num_snps)))

    # aggregate gene statistics across selected samples
    gene_counts = {}
    gene_example_func = {}

    for idx in selected_indices:
        sample_name = df_plot.loc[idx, "sample"]
        sample_data = hover_texts.get(str(sample_name)) or hover_texts.get(sample_name)
        if sample_data is None:
            continue

        top_pcs_order = sample_data.get("top_pcs", [])[:num_pcs]
        for rank, pc_idx in enumerate(top_pcs_order, start=1):
            pc_key = str(rank)
            pc_block = sample_data.get(pc_key)
            if pc_block is None:
                continue

            features = pc_block["features"][:num_snps]
            for feat in features:
                full_desc = feat[3]
                if " (" in full_desc:
                    gene, rest = full_desc.split(" (", 1)
                    gene = gene.strip()
                    func = rest.rsplit(")", 1)[0].strip()
                else:
                    gene = full_desc.strip()
                    func = ""

                if not gene:
                    continue

                gene_counts[gene] = gene_counts.get(gene, 0) + 1
                if gene not in gene_example_func and func:
                    gene_example_func[gene] = func

    if not gene_counts:
        return (
            "No gene-level interpretation available for the selected samples.",
            [],
        )

    sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)

    header = html.Thead(
        html.Tr(
            [
                html.Th("Gene", style={"textAlign": "left", "padding": "4px"}),
                html.Th("Count in selection", style={"textAlign": "left", "padding": "4px"}),
                html.Th("Example function", style={"textAlign": "left", "padding": "4px"}),
            ]
        )
    )

    rows = []
    for gene, count in sorted_genes:
        func = gene_example_func.get(gene, "")
        rows.append(
            html.Tr(
                [
                    html.Td(gene, style={"padding": "4px"}),
                    html.Td(str(count), style={"padding": "4px"}),
                    html.Td(func, style={"padding": "4px"}),
                ]
            )
        )

    body = html.Tbody(rows)

    message = (
        f"Aggregated results for {len(selected_indices)} selected samples. "
        "Top genes contributing to PCA position:"
    )

    return message, [header, body]

@app.callback(
    Output("pca-graph", "figure", allow_duplicate=True),
    Output("interpretation-message", "children", allow_duplicate=True),
    Output("interpretation-panel", "children", allow_duplicate=True),
    Input("clear-lasso-button", "n_clicks"),
    State("pca-graph", "figure"),
    State("df-plot-store", "data"),
    State("orig-marker-store", "data"),
    prevent_initial_call=True,
)
def clear_lasso(n_clicks, fig, df_json, orig):
    if not n_clicks or fig is None or df_json is None or orig is None:
        return fig, dash.no_update, dash.no_update

    df_plot = pd.read_json(df_json, orient="split")

    # reset main plot (trace 0) to base colors/sizes
    if len(fig["data"]) > 0:
        trace0 = fig["data"][0]
        trace0["marker"]["line"] = {"width": 0}
        colors = list(df_plot["Superpopulation"].map(superpop2color))
        sizes = [BASE_SIZE] * len(colors)
        trace0["marker"]["color"] = colors
        trace0["marker"]["size"] = sizes

    # reset subplots (traces 1,2,3) from orig
    for i_trace in [1, 2, 3]:
        if i_trace >= len(fig["data"]):
            continue
        key = str(i_trace)
        if key not in orig:
            continue
        trace = fig["data"][i_trace]
        trace["marker"]["line"] = {"width": 0}
        trace["marker"]["color"] = list(orig[key]["color"])
        trace["marker"]["size"] = list(orig[key]["size"])

    # reset interpretation panel to default (message above empty table)
    default_msg = (
        "Click a point or search for a sample to explore differentiating features. "
        "To explore key genes in multiple samples, use the lasso tool."
    )
    empty_header = html.Thead(
        html.Tr(
            [
                html.Th("PC", style={"textAlign": "left", "padding": "4px"}),
                html.Th("Gene", style={"textAlign": "left", "padding": "4px"}),
                html.Th("Function", style={"textAlign": "left", "padding": "4px"}),
            ]
        )
    )
    empty_body = html.Tbody([])

    return fig, default_msg, [empty_header, empty_body]


app.run(debug=True)