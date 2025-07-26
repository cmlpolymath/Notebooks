# visualization.py

import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import argparse
import duckdb
import json
from pathlib import Path

# --- Import project modules ---
import data_handler
from config import settings
from feature_engineering import FeatureCalculator

logger = logging.getLogger("flint")

# --- Helper to format large numbers ---
def format_large_number(num):
    if num is None or not isinstance(num, (int, float)): return 'N/A'
    if num > 1e12: return f"${num / 1e12:.2f} T"
    if num > 1e9: return f"${num / 1e9:.2f} B"
    if num > 1e6: return f"${num / 1e6:.2f} M"
    return f"${num:,.2f}"

def get_fuzzy_verdict(kelly_fraction: float) -> str:
    """Generates a human-readable verdict based on the Kelly fraction."""
    try:
        if kelly_fraction > 0.5:
            return "Strong Buy"
        elif kelly_fraction > 0.1:
            return "Buy"
        elif kelly_fraction < -0.5:
            return "Strong Sell"
        elif kelly_fraction < -0.1:
            return "Sell"
        else:
            return "Hold / Neutral"
    except Exception as e:
        print(f"Error determining fuzzy verdict: {e}")
        logger.exception("Error determining fuzzy verdict")
        return "Unknown"
    
# create_fundamentals_card is now crypto-aware
def create_fundamentals_card(ticker: str) -> dbc.Card | html.Div:
    """Fetches and displays key data, adapting between stocks and crypto."""
    try:
        info = data_handler.get_ticker_info(ticker)
        if not info:
            return dbc.Card(dbc.CardBody("Fundamental data not available."), color="warning", inverse=True)

        is_crypto = '-USD' in ticker.upper() or '-USDT' in ticker.upper()
        
        if is_crypto:
            card_header = "Tokenomics"
            fund_items = {
                "Market Cap": format_large_number(info.get('marketCap')),
                "Circ. Supply": format_large_number(info.get('circulatingSupply')).replace('$', ''),
                "Volume (24h)": format_large_number(info.get('volume24Hr')),
                "Algorithm": info.get('algorithm', 'N/A'),
            }
        else: # It's a stock
            card_header = "Key Fundamentals"
            fund_items = {
                "Sector": info.get('sector', 'N/A'),
                "Industry": info.get('industry', 'N/A'),
                "Fwd P/E": f"{info.get('forwardPE'):.2f}" if info.get('forwardPE') else 'N/A',
                "Beta": f"{info.get('beta'):.2f}" if info.get('beta') else 'N/A',
                "Div Yield": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
                "Inst. Own %": f"{info.get('heldPercentInstitutions', 0) * 100:.2f}%" if info.get('heldPercentInstitutions') else 'N/A',
            }
        
        list_group_items = [
            dbc.ListGroupItem([
                html.B(f"{key}:", className="flex-grow-1"),
                html.Span(value, className="ms-2 text-end")
            ], className="d-flex justify-content-between", style={'padding': '0.5rem 1rem'})
            for key, value in fund_items.items()
        ]
        
        return dbc.Card([dbc.CardHeader(card_header), dbc.ListGroup(list_group_items, flush=True)])

    except Exception as e:
        logger.exception("Error creating fundamentals card")
        return dbc.Card(dbc.CardBody(f"Error loading fundamentals: {e}"), color="danger", inverse=True)


def plot_price_with_signals(
    ticker: str,
    df: pd.DataFrame,
    test_index: pd.Index,
    predictions: pd.Series
) -> go.Figure:
    """
    Creates an advanced price chart with candlesticks, volume, MAs, and buy/sell signals.
    """
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)

        for window, color in [(50, "orange"), (200, "cyan")]:
            ma = df["Close"].rolling(window).mean()
            fig.add_trace(go.Scatter(x=df.index, y=ma, name=f"{window}D MA", line=dict(width=1, dash="dot", color=color)), row=1, col=1)

        if len(predictions) and len(test_index):
            signal_index = test_index[: len(predictions)]
            buys = signal_index[predictions == 1]
            sells = signal_index[predictions == 0]
            fig.add_trace(go.Scatter(x=buys, y=df.loc[buys, "Low"] * 0.98, mode="markers", marker_symbol="triangle-up", marker_size=12, marker_color="#00FE35", name="Predicted Buy"), row=1, col=1)
            fig.add_trace(go.Scatter(x=sells, y=df.loc[sells, "High"] * 1.02, mode="markers", marker_symbol="triangle-down", marker_size=12, marker_color="#FE0000", name="Predicted Sell"), row=1, col=1)

        volume_colors = ["#00FE35" if o <= c else "#FE0000" for o, c in zip(df["Open"], df["Close"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=volume_colors, opacity=0.7, name="Volume"), row=2, col=1)

        fig.update_layout(title_text=f"{ticker} – Price Analysis & Model Signals", template="plotly_dark", xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=20, r=20, t=50, b=20))
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        return fig
    except Exception as e:
        logger.exception("Error in plot_price_with_signals")
        err_fig = go.Figure().add_annotation(text=f"Error generating price chart:<br>{e}", showarrow=False)
        return err_fig

def plot_feature_importance(feature_importance: pd.Series) -> go.Figure:
    """Creates a horizontal bar chart for the top 10 feature importances."""
    try:
        if feature_importance.empty:
            raise ValueError("Feature importance series is empty")
        top10 = feature_importance.nlargest(10).sort_values(ascending=True)
        fig = go.Figure(go.Bar(x=top10.values, y=top10.index, orientation="h", marker=dict(colorscale="Viridis", color=top10.values, showscale=True)))
        fig.update_layout(title="Top 10 Feature Importances (Mean |SHAP|)", xaxis_title="Mean |SHAP| Value", yaxis_title="Feature", template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
        return fig
    except Exception as e:
        logger.exception("Error in plot_feature_importance")
        err_fig = go.Figure().add_annotation(text=f"Error generating importance chart:<br>{e}", showarrow=False)
        return err_fig

def plot_mc_distribution(mc_results: dict) -> go.Figure:
    """Creates a histogram of the Monte Carlo simulated trend slopes."""
    try:
        slopes = mc_results.get("simulated_slopes")
        if not slopes:
            raise ValueError("No valid simulated slopes found.")
        fig = go.Figure(data=[go.Histogram(x=slopes, nbinsx=50, name="Distribution")])
        mean_slope = np.mean(slopes)
        fig.add_vline(x=mean_slope, line_width=2, line_dash="dash", line_color="yellow", annotation_text=f"Mean: {mean_slope:.4f}")
        fig.add_vline(x=0, line_width=2, line_dash="dot", line_color="white")
        fig.update_layout(title_text="Monte Carlo: Distribution of Future Trend Slopes", xaxis_title="Simulated Slope", yaxis_title="Frequency", template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
        return fig
    except Exception as e:
        logger.exception("Error in plot_mc_distribution")
        err_fig = go.Figure().add_annotation(text=f"Error generating MC chart:<br>{e}", showarrow=False)
        return err_fig

def get_available_models(ticker: str) -> list:
    db_path = Path('results/audit_log.duckdb')
    if not db_path.exists():
        return []
    with duckdb.connect(str(db_path), read_only=True) as con:
        models_df = con.execute("SELECT DISTINCT model_name FROM analysis_results WHERE ticker = ? ORDER BY model_name DESC", [ticker]).fetchdf()
        return models_df['model_name'].tolist()

def create_app_layout(ticker: str, available_models: list):
    return dbc.Container([
        html.H1(f"Algorithmic Audit: {ticker}", className="text-center text-primary my-4"),
        dbc.Row([
            dbc.Col(html.Label("Select Model to View:"), width="auto"),
            dbc.Col(dcc.Dropdown(id='model-dropdown', options=[{'label': m, 'value': m} for m in available_models], value=available_models[0] if available_models else None, clearable=False)),
        ], className="mb-4 align-items-center"),
        html.Div(id='dashboard-content', children=[dbc.Spinner(color="primary")])
    ], fluid=True)

def build_dashboard_content(ticker, df_features, test_df, predictions, feature_importance, metrics, mc_results):
    price_fig = plot_price_with_signals(ticker, df_features, test_df.index, predictions)
    importance_fig = plot_feature_importance(feature_importance)
    mc_fig = plot_mc_distribution(mc_results)

    # --- Data Extraction from Metrics ---
    kelly = metrics.get("kelly_fraction", 0.0)
    accuracy = metrics.get("accuracy", 0.0)
    mc_strength = metrics.get("trend_strength", 0.0)
    
    # Use the verdict from the log, fall back to fuzzy logic if not present
    verdict = metrics.get("verdict") or get_fuzzy_verdict(kelly) 
    
    # Get trade levels, which could be None
    entry_price = metrics.get("entry_price")
    stop_loss = metrics.get("stop_loss")
    take_profit = metrics.get("take_profit") # Renamed from exit_price for clarity

    # --- Verdict Card Content Generation ---
    # This logic block conditionally builds the content for the verdict card
    verdict_color_class = "text-success" if "BUY" in verdict else "text-danger" if "SELL" in verdict else "text-warning"
    
    verdict_card_children = [
        html.H4("Actionable Forecast", className="card-title"),
        html.H2(verdict, className=verdict_color_class),
    ]

    # Only add the price levels if they exist (i.e., not a HOLD signal)
    if all(price is not None for price in [entry_price, stop_loss, take_profit]):
        verdict_card_children.append(
            html.P(
                f"Entry: ${entry_price:.2f} | Stop-Loss: ${stop_loss:.2f} | Target: ${take_profit:.2f}",
                className="text-muted small mt-2"
            )
        )
    else:
        verdict_card_children.append(
            html.P("No strong signal. Monitor.", className="text-muted small mt-2")
        )

    # --- Final Layout Assembly ---
    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(verdict_card_children)), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Model Certainty"), html.P(f"Kelly Bet: {kelly:.2f} | Accuracy: {accuracy:.2%}")])), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("MC Forecast"), html.P(f"Trend Strength: {mc_strength:.3f}")])), width=3),
            dbc.Col(create_fundamentals_card(ticker), width=3)
        ], className="mb-4"),
        dbc.Row(dbc.Col(dcc.Graph(figure=price_fig), width=12), className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=importance_fig), width=6),
            dbc.Col(dcc.Graph(figure=mc_fig), width=6),
        ]),
    ])

def load_run_data(ticker: str, model_name: str):
    db_path = Path('results/audit_log.duckdb')
    if not db_path.exists():
        raise FileNotFoundError("Audit DB not found. Run 'run.py' first.")

    with duckdb.connect(str(db_path), read_only=True) as con:
        result_df = con.execute("SELECT * FROM analysis_results WHERE ticker = ? AND model_name = ? ORDER BY execution_timestamp DESC LIMIT 1", [ticker, model_name]).fetchdf()
    if result_df.empty:
        raise ValueError(f"No results for ticker '{ticker}' and model '{model_name}'.")

    latest_run = result_df.iloc[0]
    metrics = json.loads(latest_run['metrics'])
    predictions_data = json.loads(latest_run['predictions'])
    shap_data = json.loads(latest_run.get('shap_importance', '{}'))
    
    feature_importance = pd.Series(data=shap_data.get('values', []), index=shap_data.get('features', []))
    
    df_raw = data_handler.get_stock_data(ticker=ticker,
                                         start_date=settings.data.start_date,
                                         end_date=settings.data.end_date)
    market_df = data_handler.get_stock_data(
        ticker=settings.data.market_index_ticker,
        start_date=settings.data.start_date,
        end_date=settings.data.end_date)
    
    df_features = FeatureCalculator(df_raw).add_all_features(market_df=market_df)
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    df_features.set_index('Date', inplace=True)
    
    df_model = df_features.copy()
    future_price = df_model['Close'].shift(-5)
    future_ma = df_model['Close'].rolling(20).mean().shift(-5)
    df_model['UpNext'] = (future_price > future_ma).astype(int)
    df_model.dropna(inplace=True)
    
    train_size = int(len(df_model) * settings.models.train_split_ratio)
    test_df = df_model.iloc[train_size:]
    
    num_predictions = len(predictions_data.get('probabilities', []))
    test_df_for_signals = test_df.iloc[-num_predictions:]

    binary_predictions = pd.Series([1 if p > 0.5 else 0 for p in predictions_data.get('probabilities', [])])
    mc_results = {k: v for k, v in metrics.items() if 'simulated_slopes' in k}

    return df_features, test_df_for_signals, binary_predictions, feature_importance, metrics, mc_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the interactive stock analysis dashboard.")
    parser.add_argument('ticker', metavar='TICKER', type=str, help='The stock ticker to visualize.')
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    
    available_models = get_available_models(ticker)
    if not available_models:
        print(f"ERROR: No models have been run for {ticker}. Run 'python run.py {ticker}' first.")
    else:
        app.layout = create_app_layout(ticker, available_models)
        @app.callback(Output('dashboard-content', 'children'), Input('model-dropdown', 'value'))
        def update_dashboard(selected_model):
            if not selected_model:
                return html.Div("Please select a model.", className="text-center")
            try:
                (df_features, test_df, predictions, fi, metrics, mc) = load_run_data(ticker, selected_model)
                return build_dashboard_content(ticker, df_features, test_df, predictions, fi, metrics, mc)
            except Exception as e:
                logger.exception(f"Error updating dashboard for {selected_model}")
                return dbc.Alert(f"Error loading data for {selected_model}: {e}", color="danger")
        print(f"Launching dashboard for {ticker}. Open http://127.0.0.1:8050 in your browser.")
        app.run(debug=True)