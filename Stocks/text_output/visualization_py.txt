# visualization.py

import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import argparse
import duckdb
import json
from pathlib import Path

# --- Import project modules to re-generate data needed for plots ---
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
        if slopes is None or len(slopes) == 0:
            raise ValueError("No simulated slopes found in Monte Carlo results.")

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

def create_dashboard(
    ticker: str,
    df_features: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: pd.Series,
    feature_importance: pd.Series,
    metrics: dict,
    mc_results: dict
) -> dash.Dash:
    """Assembles the plots and metrics into a Dash application."""
    # Build Figures with error handling
    price_fig = plot_price_with_signals(ticker, df_features, test_df.index, predictions)
    importance_fig = plot_feature_importance(feature_importance)
    mc_fig = plot_mc_distribution(mc_results)

    # Metrics
    kelly = metrics.get("kelly_fraction", 0.0)
    accuracy = metrics.get("accuracy", 0.0)
    mc_strength = metrics.get("mc_trend_strength", 0.0)
    mc_up_prob = metrics.get("mc_up_prob", 0.0)
    mc_down_prob = metrics.get("mc_down_prob", 0.0)
    mc_ci = metrics.get("mc_ci", [None, None])
    verdict = get_fuzzy_verdict(kelly)

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app.layout = dbc.Container(
        children=[
            # Header
            dbc.Row(
                dbc.Col(
                    html.H1(
                        f"Algorithmic Audit: {ticker}",
                        className="text-center text-primary mb-4"
                    ),
                    width=12
                ),
                className="mt-4"
            ),

            # Metrics
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Final Verdict", className="card-title"),
                                    html.H2(
                                        verdict,
                                        className=f"card-text {'text-success' if kelly > 0 else 'text-danger'}"
                                    ),
                                ]
                            )
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Kelly Criterion", className="card-title"),
                                    html.H2(f"{kelly:.2f}", className="card-text"),
                                ]
                            )
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Test Accuracy", className="card-title"),
                                    html.H2(f"{accuracy:.2%}", className="card-text"),
                                ]
                            )
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("MC Trend Strength", className="card-title"),
                                    html.H2(f"{metrics.get('mc_trend_strength', 0.0):.3f}", className="card-text"),
                                ]
                            )
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("MC Up Probability", className="card-title"),
                                    html.H2(f"{metrics.get('mc_up_prob', 0.0):.3f}", className="card-text"),
                                ]
                            )
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("MC Down Probability", className="card-title"),
                                    html.H2(f"{metrics.get('mc_down_prob', 0.0):.3f}", className="card-text"),
                                ]
                            )
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("MC Confidence Interval", className="card-title"),
                                    html.H2(f"[{mc_ci[0]:.3f}, {mc_ci[1]:.3f}]", className="card-text"),
                                ]
                            )
                        ),
                        width=3,
                    ),
                ],
                className="mb-4",
            ),

            # Price Chart
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id="price-chart",
                        figure=price_fig
                        ),
                     width=12
                ),
            className="mt-4"
            ),
                        
            # Bottom row for Importance and MC plots
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            id="importance-chart",
                            figure=importance_fig
                            ),
                        width=6
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id="mc-chart",
                            figure=mc_fig
                        ),
                        width=6
                    ),
                ],
                className="mt-4"
            ),
        ],
        fluid=True,
    )

    return app
def load_latest_run_data(ticker: str):
    """
    Loads all necessary data for visualization from the database and by re-running
    the data pipeline.
    """
    print(f"Loading latest run data for ticker: {ticker}")
    db_path = Path('results/audit_log.duckdb')
    if not db_path.exists():
        raise FileNotFoundError(f"Audit database not found at {db_path}. Please run 'run.py' first.")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Query for the most recent run for the given ticker
        result_df = con.execute(
            """
            SELECT * FROM analysis_results 
            WHERE ticker = ? 
            ORDER BY execution_timestamp DESC 
            LIMIT 1
            """,
            [ticker]
        ).fetchdf()
    finally:
        con.close()

    if result_df.empty:
        raise ValueError(f"No analysis results found for ticker '{ticker}' in the database.")

    # Extract data from the first (and only) row
    latest_run = result_df.iloc[0]
    
    # Parse JSON strings back into Python objects
    metrics = json.loads(latest_run['metrics'])
    predictions_data = json.loads(latest_run['predictions'])
    # run_config = json.loads(latest_run['run_config']) # We might need this later
    
    # Reconstruct feature importance Series
    # Note: This assumes 'shap_importance' and 'feature_names' were added to the log
    shap_data = json.loads(latest_run.get('shap_importance', '{}'))
    feature_importance = pd.Series(
        data=shap_data.get('values', []),
        index=shap_data.get('features', [])
    )
    # The price chart needs the full historical data, which isn't in the log.
    # We re-generate it here. This is good practice as the log only stores results.
    print("Re-generating feature data for plotting...")
    df_raw = data_handler.get_stock_data(
        ticker=ticker,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )
    df_features = FeatureCalculator(df_raw).add_all_features()
    df_features.set_index('Date', inplace=True)

    # Recreate the train/test split to find the index for plotting signals
    train_size = int(len(df_features) * config.TRAIN_SPLIT_RATIO)
    test_df = df_features.iloc[train_size:]
    
    # The logged predictions are probabilities. Convert to binary 0/1 for plotting.
    # The final prediction is an ensemble, but we can use the XGBoost proba for signals.
    xgb_proba = predictions_data.get('probabilities', [])
    binary_predictions = pd.Series([1 if p > 0.5 else 0 for p in xgb_proba])
    
    # MC results are in the metrics blob
    mc_results = {
        "simulated_slopes": metrics.get("mc_simulated_slopes")
    }

    return df_features, test_df, binary_predictions, feature_importance, metrics, mc_results

if __name__ == "__main__":
    # Use argparse to get the ticker from the command line
    parser = argparse.ArgumentParser(description="Launch the stock analysis dashboard.")
    parser.add_argument(
        'ticker',
        metavar='TICKER',
        type=str,
        help='The stock ticker to visualize (e.g., AAPL, TSLA).'
    )
    args = parser.parse_args()
    
    try:
        # 1. Load real data from the database
        df_features, test_df, predictions, feature_importance, metrics, mc_results = load_latest_run_data(args.ticker.upper())

        # 2. Create the dashboard with the loaded data
        app = create_dashboard(
            ticker=args.ticker.upper(),
            df_features=df_features,
            test_df=test_df,
            predictions=predictions,
            feature_importance=feature_importance,
            metrics=metrics,
            mc_results=mc_results
        )

        # 3. Run the Dash server
        print(f"Launching dashboard for {args.ticker.upper()}. Open http://127.0.0.1:8050 in your browser.")
        app.run_server(debug=True)

    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logger.exception("Dashboard launch failed.")