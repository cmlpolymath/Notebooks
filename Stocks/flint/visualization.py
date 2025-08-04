#!/usr/bin/env python3
"""
Modern, clean Flint visualization dashboard
Optimized for speed, browser compatibility, and aesthetics
"""
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import duckdb

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import settings

logger = logging.getLogger("flint")

# ===============================================================================
# DATA LOADING (Simplified & Fast)
# ===============================================================================

class FlintDataLoader:
    """Optimized data loader using database + parquet files instead of PyTorch tensors"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.db_path = Path('results/audit_log.duckdb')
        self.data_dir = Path('data')
        self._price_cache = None
        self._model_cache = {}
    
    def get_available_models(self) -> List[str]:
        """Get all available models for the ticker"""
        if not self.db_path.exists():
            logger.error(f"Database not found: {self.db_path}")
            return []
        
        try:
            with duckdb.connect(str(self.db_path), read_only=True) as conn:
                result = conn.execute(
                    "SELECT DISTINCT model_name FROM analysis_results WHERE ticker = ? ORDER BY execution_timestamp DESC",
                    [self.ticker]
                ).fetchall()
                return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def load_model_results(self, model_name: str) -> Optional[Dict]:
        """Load results for a specific model from database"""
        cache_key = f"{self.ticker}_{model_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        try:
            with duckdb.connect(str(self.db_path), read_only=True) as conn:
                result = conn.execute(
                    "SELECT * FROM analysis_results WHERE ticker = ? AND model_name = ? ORDER BY execution_timestamp DESC LIMIT 1",
                    [self.ticker, model_name]
                ).fetchdf()
            
            if result.empty:
                return None
            
            data = result.iloc[0].to_dict()
            self._model_cache[cache_key] = data
            return data
            
        except Exception as e:
            logger.error(f"Failed to load model results: {e}")
            return None
    
    def load_price_data(self) -> Optional[pd.DataFrame]:
        """Load price data from parquet file, fallback to processed PyTorch data"""
        if self._price_cache is not None:
            return self._price_cache
        
        # Try parquet file first (fastest and cleanest)
        parquet_path = self.data_dir / f"{self.ticker}.parquet"
        if parquet_path.exists():
            try:
                logger.info(f"Loading price data from: {parquet_path}")
                df = pd.read_parquet(parquet_path)
                
                # Ensure we have required columns for candlestick
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    self._price_cache = df
                    logger.info(f"Loaded {len(df)} price records from parquet")
                    return df
                else:
                    missing = [col for col in required_cols if col not in df.columns]
                    logger.warning(f"Parquet missing columns: {missing}")
            except Exception as e:
                logger.error(f"Failed to load parquet: {e}")
        
        # Fallback: extract from processed PyTorch data
        return self._extract_from_processed_data()
    
    def _extract_from_processed_data(self) -> Optional[pd.DataFrame]:
        """Extract basic OHLCV from processed PyTorch data"""
        try:
            logger.info("Loading price data from processed PyTorch files...")
            safe_ticker = self.ticker.replace('/', '-').replace('\\', '-')
            processed_path = Path('data/processed') / f"{safe_ticker}_data.pt"
            
            if not processed_path.exists():
                logger.error(f"Processed data file not found: {processed_path}")
                return None
            
            # Load PyTorch data with proper settings
            import torch
            torch.serialization.add_safe_globals([pd.DataFrame, pd.Series])
            data_package = torch.load(processed_path, map_location='cpu', weights_only=False)
            df = data_package.get('df_features')
            
            if df is None or df.empty:
                logger.error("No df_features found in processed data")
                return None
            
            # Extract and cache price columns
            if self._extract_price_columns(df):
                return self._price_cache
            else:
                logger.error("Failed to extract required price columns")
                return None
                    
        except Exception as e:
            logger.error(f"Failed to extract from processed data: {e}")
            return None
    
    def _extract_price_columns(self, df: pd.DataFrame) -> bool:
        """Extract and cache price columns from a processed DataFrame"""
        if df is None or df.empty:
            return False
            
        try:
            # Filter to just OHLCV columns we need
            price_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in price_cols if col in df.columns]
            
            if len(available_cols) >= 5:  # Need at least OHLC + one more
                price_df = df[available_cols].copy()
                if 'Date' in price_df.columns:
                    price_df['Date'] = pd.to_datetime(price_df['Date'])
                    price_df.set_index('Date', inplace=True)
                elif price_df.index.name != 'Date':
                    # Assume index is already datetime
                    price_df.index = pd.to_datetime(price_df.index)
                
                self._price_cache = price_df
                logger.info(f"Extracted {len(price_df)} price records from processed data")
                return True
        except Exception as e:
            logger.error(f"Failed to extract price columns: {e}")
        
        return False

# ===============================================================================
# MODERN CHART COMPONENTS
# ===============================================================================

class ModernCharts:
    """Modern, fast chart components with great aesthetics"""
    
    @staticmethod
    def create_price_chart(df: pd.DataFrame, predictions: pd.Series, ticker: str) -> go.Figure:
        """Create a beautiful, responsive price chart"""
        if df is None or df.empty:
            return ModernCharts._create_error_chart("No price data available")
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return ModernCharts._create_error_chart(f"Missing columns: {', '.join(missing_cols)}")
        
        # Prepare data
        df = df.copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.8, 0.2],
            subplot_titles=[f'{ticker} Price Action', 'Volume']
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                increasing_fillcolor='rgba(0, 255, 136, 0.3)',
                decreasing_fillcolor='rgba(255, 68, 68, 0.3)'
            ),
            row=1, col=1
        )
        
        # Moving averages
        for period, color in [(20, '#ffd700'), (50, '#ff6b35'), (200, '#4ecdc4')]:
            if len(df) >= period:
                ma = df['Close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ma,
                        name=f'{period}MA',
                        line=dict(color=color, width=1.5),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # Add predictions if available
        if predictions is not None and not predictions.empty:
            valid_predictions = predictions.dropna()
            if not valid_predictions.empty:
                # Buy signals
                buy_signals = valid_predictions[valid_predictions == 1]
                if not buy_signals.empty:
                    buy_prices = df.loc[buy_signals.index.intersection(df.index), 'Low'] * 0.98
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals.index,
                            y=buy_prices,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color='#00ff88',
                                line=dict(color='white', width=1)
                            ),
                            name='Buy Signal'
                        ),
                        row=1, col=1
                    )
                
                # Sell signals
                sell_signals = valid_predictions[valid_predictions == 0]
                if not sell_signals.empty:
                    sell_prices = df.loc[sell_signals.index.intersection(df.index), 'High'] * 1.02
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals.index,
                            y=sell_prices,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color='#ff4444',
                                line=dict(color='white', width=1)
                            ),
                            name='Sell Signal'
                        ),
                        row=1, col=1
                    )
        
        # Volume bars
        colors = ['#00ff88' if close >= open else '#ff4444' 
                 for close, open in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors,
                opacity=0.6,
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Styling
        fig.update_layout(
            template='plotly_dark',
            title=dict(
                text=f'{ticker} Analysis Dashboard',
                x=0.5,
                font=dict(size=24, color='white')
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            margin=dict(l=0, r=0, t=60, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    @staticmethod
    def create_feature_importance_chart(importance_data: Dict) -> go.Figure:
        """Create a modern feature importance chart"""
        if not importance_data or not importance_data.get('features'):
            return ModernCharts._create_error_chart("No feature importance data available")
        
        features = importance_data.get('features', [])
        values = importance_data.get('values', [])
        
        if len(features) != len(values):
            return ModernCharts._create_error_chart("Feature importance data mismatch")
        
        # Create DataFrame and sort
        df_importance = pd.DataFrame({
            'feature': features,
            'importance': values
        }).sort_values('importance', ascending=True).tail(15)  # Top 15
        
        # Create horizontal bar chart
        fig = go.Figure(
            go.Bar(
                x=df_importance['importance'],
                y=df_importance['feature'],
                orientation='h',
                marker=dict(
                    color=df_importance['importance'],
                    colorscale='Viridis',
                    colorbar=dict(title="Importance")
                ),
                text=[f'{val:.3f}' for val in df_importance['importance']],
                textposition='outside'
            )
        )
        
        fig.update_layout(
            template='plotly_dark',
            title=dict(
                text='Feature Importance (Top 15)',
                x=0.5,
                font=dict(size=20, color='white')
            ),
            xaxis_title='Importance Score',
            margin=dict(l=150, r=50, t=60, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_monte_carlo_chart(mc_data: Dict) -> go.Figure:
        """Create Monte Carlo simulation chart"""
        slopes = mc_data.get('simulated_slopes', [])
        if not slopes:
            return ModernCharts._create_error_chart("No Monte Carlo data available")
        
        # Create histogram
        fig = go.Figure(
            go.Histogram(
                x=slopes,
                nbinsx=50,
                marker=dict(
                    color='rgba(0, 255, 136, 0.7)',
                    line=dict(color='white', width=1)
                ),
                name='Distribution'
            )
        )
        
        # Add mean line
        mean_slope = np.mean(slopes)
        fig.add_vline(
            x=mean_slope,
            line_dash="dash",
            line_color="yellow",
            line_width=2,
            annotation_text=f"Mean: {mean_slope:.4f}"
        )
        
        # Add zero line
        fig.add_vline(
            x=0,
            line_dash="dot",
            line_color="white",
            line_width=1
        )
        
        fig.update_layout(
            template='plotly_dark',
            title=dict(
                text='Monte Carlo Trend Simulation',
                x=0.5,
                font=dict(size=20, color='white')
            ),
            xaxis_title='Trend Slope',
            yaxis_title='Frequency',
            margin=dict(l=50, r=50, t=60, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    
    @staticmethod
    def _create_error_chart(message: str) -> go.Figure:
        """Create a chart showing an error message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="orange")
        )
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        return fig

# ===============================================================================
# MODERN UI COMPONENTS
# ===============================================================================

def create_metric_card(title: str, value: str, color: str = "primary", icon: str = None) -> dbc.Card:
    """Create a modern metric card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H4(title, className="card-title text-muted mb-2"),
                html.H2(value, className=f"text-{color} mb-0", style={'fontWeight': 'bold'}),
            ])
        ])
    ], className="shadow-sm border-0", style={'borderRadius': '15px'})

def create_loading_spinner() -> html.Div:
    """Create a modern loading spinner"""
    return html.Div([
        dbc.Spinner(
            html.Div(),
            size="lg",
            color="primary",
            type="border",
            spinnerClassName="me-2"
        ),
        html.P("Loading analysis...", className="text-muted mt-2")
    ], className="text-center py-5")

# ===============================================================================
# MAIN APPLICATION
# ===============================================================================

def create_app(ticker: str) -> dash.Dash:
    """Create the main Dash application"""
    
    # Initialize data loader
    data_loader = FlintDataLoader(ticker)
    available_models = data_loader.get_available_models()
    
    # Create app with modern theme
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
        title=f"Flint Analysis - {ticker}"
    )
    
    # Modern layout
    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-chart-line me-3"),
                    f"Flint Analysis Dashboard"
                ], className="text-center text-primary mb-1"),
                html.H3(ticker, className="text-center text-muted mb-4")
            ])
        ]),
        
        # Model selector
        dbc.Row([
            dbc.Col([
                html.Label("Select Model:", className="form-label fw-bold"),
                dcc.Dropdown(
                    id='model-selector',
                    options=[{'label': model, 'value': model} for model in available_models],
                    value=available_models[0] if available_models else None,
                    clearable=False,
                    className="mb-4",
                    style={'borderRadius': '10px'}
                )
            ], md=6)
        ], justify="center"),
        
        # Main content area
        html.Div(id='dashboard-content', children=[create_loading_spinner()]),
        
        # Footer
        html.Hr(),
        html.P("Powered by cml_polymath", className="text-center text-muted small")
        
    ], fluid=True, className="py-4")
    
    # Main callback
    @app.callback(
        Output('dashboard-content', 'children'),
        Input('model-selector', 'value'),
        prevent_initial_call=False
    )
    def update_dashboard(selected_model):
        if not selected_model:
            return dbc.Alert("Please select a model to view results.", color="warning")
        
        try:
            # Load data
            model_results = data_loader.load_model_results(selected_model)
            price_data = data_loader.load_price_data()
            
            if not model_results:
                return dbc.Alert("No results found for selected model.", color="danger")
            
            # Parse results
            metrics = json.loads(model_results.get('metrics', '{}'))
            predictions_data = json.loads(model_results.get('predictions', '{}'))
            shap_data = json.loads(model_results.get('shap_importance', '{}'))
            
            # Prepare predictions
            predictions = None
            if predictions_data.get('dates') and predictions_data.get('probabilities'):
                pred_dates = pd.to_datetime(predictions_data['dates'])
                pred_values = [1 if p > 0.5 else 0 for p in predictions_data['probabilities']]
                predictions = pd.Series(pred_values, index=pred_dates)
            
            # Create charts
            price_chart = ModernCharts.create_price_chart(price_data, predictions, ticker)
            importance_chart = ModernCharts.create_feature_importance_chart(shap_data)
            mc_chart = ModernCharts.create_monte_carlo_chart(metrics)
            
            # Extract key metrics
            verdict = metrics.get('verdict', 'N/A')
            kelly = metrics.get('kelly_fraction', 0.0)
            accuracy = metrics.get('accuracy', 0.0)
            trend_strength = metrics.get('trend_strength', 0.0)
            
            verdict_color = "success" if "BUY" in verdict else "danger" if "SELL" in verdict else "warning"
            
            # Build dashboard
            return html.Div([
                # Metrics row
                dbc.Row([
                    dbc.Col(create_metric_card("Signal", verdict, verdict_color), md=3),
                    dbc.Col(create_metric_card("Kelly Fraction", f"{kelly:.3f}"), md=3),
                    dbc.Col(create_metric_card("Accuracy", f"{accuracy:.1%}"), md=3),
                    dbc.Col(create_metric_card("Trend Strength", f"{trend_strength:.3f}"), md=3),
                ], className="mb-4"),
                
                # Main chart
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            figure=price_chart,
                            config={'displayModeBar': True, 'responsive': True}
                        )
                    ])
                ], className="mb-4"),
                
                # Analysis charts
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            figure=importance_chart,
                            config={'displayModeBar': False, 'responsive': True}
                        )
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(
                            figure=mc_chart,
                            config={'displayModeBar': False, 'responsive': True}
                        )
                    ], md=6)
                ])
            ])
            
        except Exception as e:
            logger.exception(f"Error updating dashboard: {e}")
            return dbc.Alert([
                html.H4("Error Loading Dashboard", className="alert-heading"),
                html.P(f"An error occurred: {str(e)}"),
                html.P("Please check the logs for more details.", className="mb-0")
            ], color="danger")
    
    return app

# ===============================================================================
# MAIN ENTRY POINT
# ===============================================================================

if __name__ == '__main__':
    settings.configure_logging()
    parser = argparse.ArgumentParser(description='Flint Modern Visualization Dashboard')
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--port', type=int, default=8050, help='Port to run on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Flint dashboard for {args.ticker}")
    
    app = create_app(args.ticker.upper())
    
    print(f"""
🚀 Flint Analytics Dashboard Starting...
📊 Ticker: {args.ticker.upper()}
🌐 URL: http://{args.host}:{args.port}
🔧 Debug: {args.debug}

Open your browser and navigate to the URL above!
    """)
    
    app.run(
        debug=args.debug,
        host=args.host,
        port=args.port,
        dev_tools_ui=args.debug,
        dev_tools_hot_reload=args.debug
    )