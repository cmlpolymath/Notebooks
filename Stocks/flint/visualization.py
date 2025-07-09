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
import config
from feature_engineering import FeatureCalculator

# ─── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def plot_price_with_signals(
    ticker: str,
    df: pd.DataFrame,
    test_index: pd.Index,
    predictions: pd.Series
) -> go.Figure:
    """
    Creates an advanced price chart with candlesticks, volume, MAs, and buy/sell signals.
    Falls back to an error annotation if anything goes wrong.
    """
    try:
        # Input validation
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"DataFrame is missing columns: {missing}")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.8, 0.2]
        )

        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Price"
            ), row=1, col=1
        )

        # Moving Averages
        for window, color, dash in [(50, "orange", "dot"), (200, "cyan", "dot")]:
            ma = df["Close"].rolling(window).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=ma,
                    name=f"{window}-Day MA",
                    line=dict(width=1, dash=dash, color=color)
                ),
                row=1, col=1
            )

        # Buy/Sell signals
        if len(predictions) and len(test_index):
            signal_index = test_index[: len(predictions)]
            signals = pd.Series(predictions, index=signal_index)
            buys = signals[signals == 1]
            sells = signals[signals == 0]

            if not buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buys.index,
                        y=df.loc[buys.index, "Low"] * 0.98,
                        mode="markers",
                        marker_symbol="triangle-up",
                        marker_size=12,
                        marker_color="#00FE35",
                        name="Predicted Buy"
                    ), row=1, col=1
                )
            if not sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sells.index,
                        y=df.loc[sells.index, "High"] * 1.02,
                        mode="markers",
                        marker_symbol="triangle-down",
                        marker_size=12,
                        marker_color="#FE0000",
                        name="Predicted Sell"
                    ), row=1, col=1
                )

        # Volume bars
        volume_colors = [
            "#00FE35" if o <= c else "#FE0000"
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=volume_colors,
                opacity=0.7,
                name="Volume"
            ),
            row=2, col=1
        )

        fig.update_layout(
            title_text=f"{ticker} – Price Analysis & Model Signals",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        return fig

    except Exception as e:
        logger.exception("Error in plot_price_with_signals")
        # Fallback empty figure with error text
        err_fig = go.Figure()
        err_fig.add_annotation(
            text=f"Error generating price chart:<br>{e}",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="red")
        )
        return err_fig

def plot_feature_importance(feature_importance: pd.Series) -> go.Figure:
    """Creates a horizontal bar chart for the top 10 feature importances."""
    try:
        if feature_importance.empty:
            raise ValueError("Feature importance series is empty")

        top10 = feature_importance.nlargest(10).sort_values(ascending=True)

        fig = go.Figure(
            go.Bar(
                x=top10.values,
                y=top10.index,
                orientation="h",
                marker=dict(colorscale="Viridis", color=top10.values, showscale=True)
            )
        )
        fig.update_layout(
            title="Top 10 Feature Importances (Mean |SHAP|)",
            xaxis_title="Mean |SHAP| Value",
            yaxis_title="Feature",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    except Exception as e:
        logger.exception("Error in plot_feature_importance")
        err_fig = go.Figure()
        err_fig.add_annotation(
            text=f"Error generating feature importance chart:<br>{e}",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="red")
        )
        return err_fig

def plot_mc_distribution(mc_results: dict) -> go.Figure:
    """Creates a histogram of the Monte Carlo simulated trend slopes."""
    try:
        slopes = mc_results.get("simulated_slopes")
        if slopes is None or not isinstance(slopes, list) or len(slopes) == 0:
            raise ValueError("No valid simulated slopes found in Monte Carlo results.")

        fig = go.Figure(data=[go.Histogram(x=slopes, nbinsx=50, name="Distribution")])
        
        mean_slope = np.mean(slopes)
        fig.add_vline(x=mean_slope, line_width=2, line_dash="dash", line_color="yellow",
                      annotation_text=f"Mean Slope: {mean_slope:.4f}")
        fig.add_vline(x=0, line_width=2, line_dash="dot", line_color="white")

        fig.update_layout(
            title_text="Monte Carlo Simulation: Distribution of Future Trend Slopes",
            xaxis_title="Simulated Slope (Price Change per Day)",
            yaxis_title="Frequency",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig
    except Exception as e:
        logger.exception("Error in plot_mc_distribution")
        err_fig = go.Figure()
        err_fig.add_annotation(text=f"Error generating MC chart:<br>{e}", showarrow=False)
        return err_fig

def get_available_models(ticker: str) -> list:
    """Queries the database to find all unique model_name entries for a ticker."""
    db_path = Path('results/audit_log.duckdb')
    if not db_path.exists():
        return []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        models_df = con.execute(
            "SELECT DISTINCT model_name FROM analysis_results WHERE ticker = ? ORDER BY model_name DESC",
            [ticker]
        ).fetchdf()
        return models_df['model_name'].tolist()
    finally:
        con.close()

def create_app_layout(ticker: str, available_models: list):
    """Creates the Dash app layout with a dropdown for model selection."""
    return dbc.Container(
        [
            html.H1(f"Algorithmic Audit: {ticker}", className="text-center text-primary my-4"),
            dbc.Row([
                dbc.Col(html.Label("Select Model to View:"), width="auto"),
                dbc.Col(
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=[{'label': m, 'value': m} for m in available_models],
                        value=available_models[0] if available_models else None,
                        clearable=False
                    ),
                ),
            ], className="mb-4 align-items-center"),
            # This Div will be populated by the callback
            html.Div(id='dashboard-content', children=[
                dbc.Spinner(color="primary") # Show a spinner on initial load
            ])
        ],
        fluid=True
    )

def build_dashboard_content(ticker, df_features, test_df, predictions, feature_importance, metrics, mc_results):
    """Builds the actual charts and metrics cards for the dashboard."""
    price_fig = plot_price_with_signals(ticker, df_features, test_df.index, predictions)
    importance_fig = plot_feature_importance(feature_importance)
    mc_fig = plot_mc_distribution(mc_results)

    kelly = metrics.get("kelly_fraction", 0.0)
    accuracy = metrics.get("accuracy", 0.0)
    mc_strength = metrics.get("trend_strength", 0.0) # Corrected key
    verdict = get_fuzzy_verdict(kelly)

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Final Verdict"), html.H2(verdict, className=f"text-{'success' if kelly > 0 else 'danger'}")])), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("Model Certainty"), html.P(f"Kelly Bet: {kelly:.2f} | Accuracy: {accuracy:.2%}")])), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4("MC Forecast"), html.P(f"Trend Strength: {mc_strength:.3f}")])), width=4),
        ], className="mb-4"),
        dbc.Row(dbc.Col(dcc.Graph(figure=price_fig), width=12), className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=importance_fig), width=6),
            dbc.Col(dcc.Graph(figure=mc_fig), width=6),
        ]),
    ])

def load_run_data(ticker: str, model_name: str):
    """Loads all necessary data for visualization from the database for a specific model run."""
    print(f"Loading run data for ticker: {ticker}, model: {model_name}")
    db_path = Path('results/audit_log.duckdb')
    if not db_path.exists():
        raise FileNotFoundError(f"Audit database not found at {db_path}. Please run 'run.py' first.")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result_df = con.execute(
            "SELECT * FROM analysis_results WHERE ticker = ? AND model_name = ? ORDER BY execution_timestamp DESC LIMIT 1",
            [ticker, model_name]
        ).fetchdf()
    finally:
        con.close()

    if result_df.empty:
        raise ValueError(f"No results found for ticker '{ticker}' and model '{model_name}'.")

    latest_run = result_df.iloc[0]
    metrics = json.loads(latest_run['metrics'])
    predictions_data = json.loads(latest_run['predictions'])
    shap_data = json.loads(latest_run.get('shap_importance', '{}'))
    
    feature_importance = pd.Series(
        data=shap_data.get('values', []),
        index=shap_data.get('features', [])
    )
    
    print("Re-generating feature data for plotting...")
    df_raw = data_handler.get_stock_data(ticker=ticker, start_date=config.START_DATE, end_date=config.END_DATE)
    market_df = data_handler.get_stock_data(ticker=config.MARKET_INDEX_TICKER, start_date=config.START_DATE, end_date=config.END_DATE)
    
    # --- THE FIX IS HERE ---
    # Do NOT set the index before passing to FeatureCalculator.
    # It expects 'Date' to be a column.
    
    df_features = FeatureCalculator(df_raw).add_all_features(market_df=market_df)
    
    # Now that features are calculated, set the index for plotting and slicing.
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    df_features.set_index('Date', inplace=True)
    
    # Recreate the train/test split to find the index for plotting signals
    df_model = df_features.copy()
    
    future_price = df_model['Close'].shift(-5)
    future_ma = df_model['Close'].rolling(20).mean().shift(-5)
    df_model['UpNext'] = (future_price > future_ma).astype(int)
    df_model.dropna(inplace=True)
    
    train_size = int(len(df_model) * config.TRAIN_SPLIT_RATIO)
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
        print(f"ERROR: No models have been run for {ticker}. Please run 'python run.py {ticker}' first.")
    else:
        app.layout = create_app_layout(ticker, available_models)

        @app.callback(
            Output('dashboard-content', 'children'),
            Input('model-dropdown', 'value')
        )
        def update_dashboard(selected_model):
            if not selected_model:
                return html.Div("Please select a model to view the analysis.", className="text-center")
            try:
                (df_features, test_df, predictions, feature_importance, metrics, mc_results) = load_run_data(ticker, selected_model)
                return build_dashboard_content(ticker, df_features, test_df, predictions, feature_importance, metrics, mc_results)
            except Exception as e:
                logger.exception(f"Error updating dashboard for {selected_model}")
                return html.Div(f"An error occurred while loading data for {selected_model}: {e}", className="alert alert-danger")

        print(f"Launching dashboard for {ticker}. Open http://127.0.0.1:8050 in your browser.")
        app.run(debug=True)