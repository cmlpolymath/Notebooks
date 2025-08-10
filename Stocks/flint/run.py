# run.py
import argparse
import json
from pathlib import Path
import time
import re
import numpy as np
import pandas as pd
import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Import project modules
from config import settings
import audit_logger
import models
import predictors
from preprocess import prepare_and_cache_data

# Instantiate the logger. It will automatically use the config from settings.
logger = structlog.get_logger(__name__)
console = Console()

def validate_ticker(ticker: str) -> str:
    """Sanitize and validate a stock or crypto ticker format."""
    ticker = ticker.strip().upper()
    if re.match(r'^[A-Z0-9\.\-]{1,12}$', ticker):
        return ticker
    raise argparse.ArgumentTypeError(f"'{ticker}' is not a valid ticker format.")

def main():
    parser = argparse.ArgumentParser(description="Run the stock/crypto analysis pipeline.")
    parser.add_argument('tickers', metavar='TICKER', type=validate_ticker, nargs='+',
                        help='A space-separated list of tickers to analyze.')
    parser.add_argument(
        '--model', '-m',
        metavar='MODEL',
        type=str, 
        default='ensemble', 
        choices=['ensemble', 'xgb', 'rf'],
        help="Model to run: 'ensemble', 'xgb' (XGBoost only), or 'rf' (Random Forest only)."
    )
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help="Force reprocessing of data, ignoring any existing cached .pt files."
    )

    parser.add_argument(
        '--profile',
        action='store_true',
        help="Run in profiling mode. This disables some optimizations like torch.compile and DataLoader workers to ensure accurate profiling."
    )

    args = parser.parse_args()

    if args.profile:
        logger.warn((f"!!! RUNNING IN PROFILING MODE. PERFORMANCE WILL BE REDUCED!!!"))
        settings.system.enable_profiling = True
    
    audit_logger.setup_audit_db()

    for ticker in args.tickers:
        # Bind context for this specific run. All logs within this loop will have these tags.
        log = logger.bind(ticker=ticker, model=args.model)
        log.info("processing_start", tickers=args.tickers)
        
        try:
            # --- UNIFIED PARAMETER LOADING ---
            tuned_params = {}
            params_path = Path("results") / "tuning" / f"best_params_{ticker}.json"
            if params_path.exists():
                log.info("tuned_params_found", path=str(params_path))
                with params_path.open("r") as f:
                    tuned_params = json.load(f)
            
            tuned_xgb_params = tuned_params.get("xgboost")
            tuned_transformer_params = tuned_params.get("transformer")

            # Step 1: Load Data
            log.info("data_preparation_start")
            data_package = prepare_and_cache_data(
                ticker, settings.data.start_date, settings.data.end_date,
                force_reprocess=args.force_reprocess
            )
            X_train, y_train = data_package['X_train'], data_package['y_train']
            X_test, y_test_orig = data_package['X_test'], data_package['y_test_orig']
            feature_cols, df_features = data_package['feature_cols'], data_package['df_features']
            log.info("data_preparation_complete", train_shape=X_train.shape, test_shape=X_test.shape)

            # Step 2: Run Monte Carlo simulation
            log.info("monte_carlo_start")
            mc_final_signal = {}
            try:
                mc_params = settings.models.monte_carlo.model_dump(exclude_unset=True)
                mc_filter = predictors.MonteCarloTrendFilter(**mc_params)
                mc_filter.fit(df_features) 
                
                mc_final_signal = mc_filter.predict(horizon=21)
                
                trend_strength = mc_final_signal.get('trend_strength', 0.0)
                if not isinstance(trend_strength, (float, int)):
                    trend_strength = 0.0

                # force it to a float and format with a leading + or – and three decimals
                signed_ts = f"{float(trend_strength):+.3f}"

                log.info(
                    "monte_carlo_complete",
                    trend_strength=signed_ts
                )

            except Exception as e:
                log.error("monte_carlo_failed", error=str(e))

            # --- Step 3: MODEL SELECTION & TRAINING ---
            final_proba = np.array([])
            model_name_log = ""
            shap_values = np.zeros(len(feature_cols))
            run_config = {}

            if args.model == 'rf':
                log.info("training_start", component="RandomForest")
                rf_model, scaler, selector, run_config = models.train_enhanced_random_forest(
                    X_train, y_train
                )
                
                # Maintain DataFrame structure after scaling
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
                
                # Maintain DataFrame structure after feature selection
                # Get the selected feature names from the selector
                selected_feature_mask = selector.get_support()
                selected_feature_names = X_test.columns[selected_feature_mask].tolist()
                
                X_test_selected = pd.DataFrame(
                    selector.transform(X_test_scaled),
                    columns=selected_feature_names,
                    index=X_test_scaled.index
                )

                final_proba = rf_model.predict_proba(X_test_selected)[:, 1]
                y_test_final = y_test_orig.values
                model_name_log = run_config["model_type"] 

            elif args.model == 'xgb':
                log.info("training_start", component="XGBoost", tuned=bool(tuned_xgb_params))
                if tuned_xgb_params:
                    log.info("Using tuned XGBoost parameters.", component="XGBoost")
                    xgb_params_to_use = tuned_xgb_params
                else:
                    log.info("No tuned xgb params. Using default", component="XGBoost")
                    xgb_params_to_use = settings.models.xgboost.model_dump(exclude_unset=True)
                xgb_model = models.train_xgboost(X_train, y_train, params=xgb_params_to_use)
                final_proba = xgb_model.predict_proba(X_test)[:, 1]
                y_test_final = y_test_orig.values
                shap_values = models.get_shap_importance(xgb_model, X_test)
                model_name_log = "XGBoost_v2_Tuned" if tuned_xgb_params else "XGBoost_v2_Default"
                run_config = {"model_type": "xgb", "xgboost_params": xgb_params_to_use}
                log.info("training_complete", component="XGBoost")

            elif args.model == 'ensemble':
                log.info("training_start", component="Ensemble")
                # XGBoost Component
                log.info("training_phase_1_xgb", sub_component="XGBoost", tuned=bool(tuned_xgb_params))
                if tuned_xgb_params:
                    log.info("Using tuned XGBoost params.", sub_component="XGBoost")
                    xgb_params_to_use = tuned_xgb_params
                else:
                    log.info("No tuned params found. Using default", sub_component="XGBoost")
                    xgb_params_to_use = settings.models.xgboost.model_dump(exclude_unset=True)
                xgb_model = models.train_xgboost(X_train, y_train, params=xgb_params_to_use)
                xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
                shap_values = models.get_shap_importance(xgb_model, X_test)
                log.info("training_phase_1_complete", sub_component="XGBoost")
                # Transformer Component
                shap_series = pd.Series(shap_values, index=feature_cols)
                top_15_features = shap_series.nlargest(15).index.tolist()
                log.info("feature_selection_for_transformer", top_features=top_15_features)
                X_train_reduced, X_test_reduced = X_train[top_15_features], X_test[top_15_features]
                log.info("training_start_phase_2_start", sub_component="Transformer", tuned=bool(tuned_transformer_params))
                if tuned_transformer_params:
                    log.info("Using tuned Transformer params.", sub_component="Transformer")
                    # Start with default training params and override/add the tuned ones
                    trans_training_params = settings.models.transformer_training.model_dump(exclude_unset=True)
                    trans_training_params.update(tuned_transformer_params)
                else:
                    log.info("No tuned params found. Using default", sub_component="Transformer")
                    trans_training_params = settings.models.transformer_training.model_dump(exclude_unset=True)
                trans_model, device, scaler, y_seq_test = models.train_transformer(
                    X_train_reduced.values, y_train.values, X_test_reduced.values, y_test_orig.values,
                    training_params=trans_training_params
                )
                trans_proba = models.predict_transformer(trans_model, device, scaler, X_test_reduced.values)[:, 1]
                log.info("training_phase_2_complete", sub_component="Transformer")
                # Ensemble Prediction
                log.info("generating_ensemble_prediction", component="Ensemble")
                min_len = min(len(xgb_proba), len(trans_proba))
                final_proba = (xgb_proba[:min_len] + trans_proba[:min_len]) / 2
                y_test_final = y_seq_test
                model_name_log = "Ensemble_v3_Tuned" if tuned_xgb_params or tuned_transformer_params else "Ensemble_v3_Default"
                run_config = {"model_type": "ensemble", "xgboost_params": xgb_params_to_use, "transformer_training_params": trans_training_params}
                log.info("training_complete", component="Ensemble")

            # --- Step 4: Common Post-Processing & Logging ---
            predictions = (final_proba > 0.500).astype(int)
            accuracy = np.mean(predictions == y_test_final) if len(y_test_final) > 0 else 0.0
            kelly_fraction = 2 * final_proba[-1] - 1 if len(final_proba) > 0 else 0.0

            # --- Actionable Forecast Calculation ---
            verdict = "HOLD / NEUTRAL"
            entry_price, stop_loss, take_profit = None, None, None
            confidence_threshold = 0.10

            if len(final_proba) > 0:
                last_close = df_features['Close'].iloc[-1]
                last_atr = df_features['ATR14'].iloc[-1]
                
                if predictions[-1] == 1 and kelly_fraction > confidence_threshold:
                    verdict = "BUY"
                    entry_price, stop_loss, take_profit = last_close, last_close - (1.5 * last_atr), last_close + (3.0 * last_atr)
                elif predictions[-1] == 0 and kelly_fraction < -confidence_threshold:
                    verdict = "SELL (SHORT)"
                    entry_price, stop_loss, take_profit = last_close, last_close + (1.5 * last_atr), last_close - (3.0 * last_atr)

            # --- Logging and Presentation ---
            # 1. Log the key metrics in a structured format for machine readability.
            log.info(
                "forecast_summary", 
                verdict=verdict,
                kelly=f"{kelly_fraction:.3f}",
                accuracy=f"{accuracy:.3%}"
            )

            # 2. Create and print a beautiful, human-readable table for the console.

            forecast_table = Table(
                show_header=True,
                header_style="bold white on dark_blue",
                border_style="bright_blue",
                box=box.HEAVY_EDGE,
                padding=(0, 1),
                caption="Generated by Project Flint • Data-driven insights ⚡"
            )
            forecast_table.add_column("Metric", style="bold cyan", justify="right", no_wrap=True)
            forecast_table.add_column("Value", style="white", justify="left")

            def add_styled_row(table, label, value, idx):
                style = "dim" if idx % 2 == 0 else ""
                table.add_row(label, value, style=style)

            rows = []
            verdict_style = "green" if "BUY" in verdict else "red" if "SELL" in verdict else "yellow"
            rows.append(("Verdict:", f"[{verdict_style}]{verdict}[/{verdict_style}]"))
            rows.append(("Last Close:", f"${last_close:.2f}"))
            if verdict != "HOLD / NEUTRAL":
                rows.extend([
                    ("Entry Price:", f"~${entry_price:.2f}"),
                    ("Stop-Loss:",   f"~${stop_loss:.2f}"),
                    ("Take-Profit:", f"~${take_profit:.2f}"),
                ])

            for idx, (label, val) in enumerate(rows):
                add_styled_row(forecast_table, label, val, idx)

            panel = Panel(
                forecast_table,
                title="[bold magenta]📈 Trading Recommendation[/]",
                border_style="magenta",
                box=box.ROUNDED
            )
            
            console.print(panel)

            # --- Metrics Assembly for Database Logging ---
            num_predictions = len(final_proba)
            test_index_for_log = pd.DatetimeIndex(data_package['test_df']['Date'].tail(num_predictions))

            metrics = {
                "accuracy": accuracy, "kelly_fraction": kelly_fraction, 'verdict': verdict,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                **mc_final_signal
            }

            audit_logger.log_analysis_result(
                ticker=ticker, 
                model_name=model_name_log, 
                run_config=run_config,
                predictions={"probabilities": final_proba.tolist()}, 
                test_index=test_index_for_log,
                metrics=metrics,
                shap_importance={'features': feature_cols, 'values': shap_values.tolist()}
            )
            log.info("processing_complete", results_logged=True)

        except Exception as e:
            log.error("processing_failed", error=str(e), exc_info=True)
            continue

if __name__ == '__main__':
    # Configure logging once at the very start of the application
    settings.configure_logging()
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    logger.info("total_execution_time", duration_seconds=round(end_time - start_time, 3))