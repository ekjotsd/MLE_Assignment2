"""
Visualization Script
Creates charts for model performance and stability over time
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from pipeline_config.config import (
    GOLD_PATH, MODEL_STORE_PATH, RESULTS_PATH,
    MODEL_MONITORING_FILE, MONITORING_THRESHOLDS,
    MONITOR_START_DATE, MONITOR_END_DATE
)

# Set custom style - Professional theme
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
# Custom color palette - Purple/Teal theme
CUSTOM_COLORS = {
    'primary': '#6A0DAD',    # Purple
    'secondary': '#20B2AA',  # Teal
    'warning': '#FF8C00',    # Dark Orange
    'danger': '#DC143C',     # Crimson
    'success': '#2E8B57',    # Sea Green
    'neutral': '#4B0082'     # Indigo
}


def load_monitoring_history():
    """
    Load monitoring history from cumulative monitoring file
    
    Returns:
        DataFrame with monitoring history
    """
    monitoring_file = MODEL_STORE_PATH / MODEL_MONITORING_FILE
    
    if not monitoring_file.exists():
        print(f"No monitoring history found at: {monitoring_file}")
        return None
    
    with open(monitoring_file, 'r') as f:
        data = json.load(f)
    
    if 'monitoring_history' not in data or len(data['monitoring_history']) == 0:
        print("Monitoring history is empty")
        return None
    
    # Convert to DataFrame
    records = []
    for record in data['monitoring_history']:
        row = {
            'snapshot_date': record['snapshot_date'],
            'model_name': record['model_name'],
            'monitored_at': record['monitored_at']
        }
        row.update(record['metrics'])
        row.update({f'check_{k}': v for k, v in record['threshold_checks'].items()})
        records.append(row)
    
    df = pd.DataFrame(records)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df = df.sort_values('snapshot_date')

    # Filter to configured monitoring period
    try:
        start_dt = pd.to_datetime(MONITOR_START_DATE)
        end_dt = pd.to_datetime(MONITOR_END_DATE)
        before_len = len(df)
        df = df[(df['snapshot_date'] >= start_dt) & (df['snapshot_date'] <= end_dt)]
        after_len = len(df)
        if after_len == 0:
            print(f"Warning: No monitoring records within configured range {MONITOR_START_DATE} to {MONITOR_END_DATE}")
        else:
            print(f"Filtered monitoring records to configured range: {MONITOR_START_DATE} to {MONITOR_END_DATE} (kept {after_len}/{before_len})")
    except Exception as e:
        print(f"Warning: could not apply monitoring date filter: {e}")
    
    print(f"Loaded monitoring history: {len(df)} records")
    print(f"Date range: {df['snapshot_date'].min()} to {df['snapshot_date'].max()}\n")
    
    return df


def plot_performance_metrics(df, output_dir):
    """
    Plot performance metrics over time with AREA CHARTS and custom styling
    CUSTOMIZATION: Changed from line plots to filled area charts
    
    Args:
        df: DataFrame with monitoring history
        output_dir: Directory to save plots
    """
    # CUSTOM: 3x2 layout instead of 2x3 for better vertical flow
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Model Performance Tracking Dashboard', fontsize=18, fontweight='bold', 
                 color=CUSTOM_COLORS['primary'])
    
    metrics = ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score', 'log_loss']
    thresholds = {
        'auc_roc': MONITORING_THRESHOLDS['auc_roc_min'],
        'precision': MONITORING_THRESHOLDS['precision_min'],
        'recall': MONITORING_THRESHOLDS['recall_min'],
        'f1_score': MONITORING_THRESHOLDS['f1_score_min']
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric in df.columns:
            # CUSTOM: Area plot with fill instead of simple line
            ax.plot(df['snapshot_date'], df[metric], 
                   marker='D', linewidth=2.5, markersize=7, 
                   color=CUSTOM_COLORS['primary'], label=metric.upper())
            ax.fill_between(df['snapshot_date'], df[metric], alpha=0.3, 
                           color=CUSTOM_COLORS['secondary'])
            
            # CUSTOM: Add value annotations on last point
            last_val = df[metric].iloc[-1]
            ax.annotate(f'{last_val:.3f}', 
                       xy=(df['snapshot_date'].iloc[-1], last_val),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       fontsize=9, fontweight='bold')
            
            # Add threshold line with custom color
            if metric in thresholds:
                ax.axhline(y=thresholds[metric], color=CUSTOM_COLORS['danger'], 
                          linestyle=':', linewidth=2, 
                          label=f'Min Threshold: {thresholds[metric]:.2f}')
            
            ax.set_xlabel('Monitoring Period', fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_title(f'📊 {metric.replace("_", " ").upper()}', fontsize=12, 
                        fontweight='bold', color=CUSTOM_COLORS['neutral'])
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / 'performance_metrics_over_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def plot_psi_over_time(df, output_dir):
    """
    Plot PSI (Population Stability Index) over time
    
    Args:
        df: DataFrame with monitoring history
        output_dir: Directory to save plots
    """
    if 'psi' not in df.columns:
        print("PSI data not available, skipping PSI plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot PSI with custom color
    ax.plot(df['snapshot_date'], df['psi'], marker='o', linewidth=2, markersize=6, 
            color=CUSTOM_COLORS['primary'], label='PSI')
    
    # Add threshold lines with custom colors
    ax.axhline(y=MONITORING_THRESHOLDS['psi_warning'], color=CUSTOM_COLORS['warning'], linestyle='--', 
              linewidth=1.5, label=f"Warning ({MONITORING_THRESHOLDS['psi_warning']})")
    ax.axhline(y=MONITORING_THRESHOLDS['psi_critical'], color=CUSTOM_COLORS['danger'], linestyle='--', 
              linewidth=1.5, label=f"Critical ({MONITORING_THRESHOLDS['psi_critical']})")
    
    # Add colored background zones with custom palette
    ax.axhspan(0, MONITORING_THRESHOLDS['psi_warning'], alpha=0.1, color=CUSTOM_COLORS['success'], label='Stable')
    ax.axhspan(MONITORING_THRESHOLDS['psi_warning'], MONITORING_THRESHOLDS['psi_critical'], 
              alpha=0.1, color=CUSTOM_COLORS['warning'], label='Moderate Drift')
    ax.axhspan(MONITORING_THRESHOLDS['psi_critical'], ax.get_ylim()[1], 
              alpha=0.1, color=CUSTOM_COLORS['danger'], label='Significant Drift')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('PSI Value')
    ax.set_title('Population Stability Index (PSI) Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / 'psi_over_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_confusion_matrix_trend(df, output_dir):
    """
    Plot confusion matrix metrics over time with STACKED BAR CHART
    CUSTOMIZATION: Changed from line plots to stacked bar chart for better composition view
    
    Args:
        df: DataFrame with monitoring history
        output_dir: Directory to save plots
    """
    # CUSTOM: Single large plot with stacked bars + line subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('📈 Prediction Composition Analysis', fontsize=16, fontweight='bold',
                 color=CUSTOM_COLORS['primary'])
    
    metrics = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']
    # CUSTOM: Custom color palette for confusion matrix
    cm_colors = [CUSTOM_COLORS['success'], CUSTOM_COLORS['warning'], 
                 CUSTOM_COLORS['secondary'], CUSTOM_COLORS['danger']]
    
    # CUSTOM: Top chart - STACKED BAR instead of line plots
    if all(m in df.columns for m in metrics):
        dates = df['snapshot_date']
        bottom = np.zeros(len(df))
        
        for metric, color in zip(metrics, cm_colors):
            values = df[metric].values
            ax1.bar(dates, values, bottom=bottom, label=metric.replace('_', ' ').title(),
                   color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
            bottom += values
        
        ax1.set_xlabel('Monitoring Period', fontweight='bold')
        ax1.set_ylabel('Prediction Count', fontweight='bold')
        ax1.set_title('Stacked Composition of Predictions', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', framealpha=0.95, ncol=2)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # CUSTOM: Bottom chart - Accuracy trend with markers
        if 'accuracy' in df.columns:
            ax2.plot(dates, df['accuracy'], marker='s', linewidth=2.5, markersize=8,
                    color=CUSTOM_COLORS['primary'], label='Accuracy', linestyle='-')
            ax2.fill_between(dates, df['accuracy'], alpha=0.2, color=CUSTOM_COLORS['primary'])
            ax2.axhline(y=0.7, color=CUSTOM_COLORS['danger'], linestyle='--', 
                       linewidth=2, label='Min Threshold (0.70)')
            ax2.set_xlabel('Monitoring Period', fontweight='bold')
            ax2.set_ylabel('Accuracy Score', fontweight='bold')
            ax2.set_title('Model Accuracy Trend', fontsize=13, fontweight='bold')
            ax2.legend(loc='best', framealpha=0.95)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / 'confusion_matrix_trend.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def plot_threshold_compliance(df, output_dir):
    """
    Plot threshold compliance over time
    
    Args:
        df: DataFrame with monitoring history
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    checks = ['check_auc_roc_check', 'check_precision_check', 'check_recall_check', 'check_f1_score_check']
    check_labels = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
    
    # Calculate compliance rate for each date
    compliance_data = []
    for _, row in df.iterrows():
        passed = sum([row.get(check, False) for check in checks])
        total = len(checks)
        compliance_rate = (passed / total) * 100
        compliance_data.append({
            'snapshot_date': row['snapshot_date'],
            'compliance_rate': compliance_rate,
            'passed_checks': passed,
            'total_checks': total
        })
    
    compliance_df = pd.DataFrame(compliance_data)
    
    # Plot compliance rate with custom colors
    ax.plot(compliance_df['snapshot_date'], compliance_df['compliance_rate'], 
           marker='o', linewidth=2, markersize=8, color=CUSTOM_COLORS['secondary'])
    ax.axhline(y=100, color=CUSTOM_COLORS['success'], linestyle='--', linewidth=1.5, label='100% Compliance')
    ax.axhline(y=75, color=CUSTOM_COLORS['warning'], linestyle='--', linewidth=1.5, label='75% Compliance')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Compliance Rate (%)')
    ax.set_title('Model Threshold Compliance Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / 'threshold_compliance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_prediction_distribution(output_dir):
    """
    Plot distribution of predictions over time
    
    Args:
        output_dir: Directory to save plots
    """
    predictions_dir = GOLD_PATH / "predictions"
    prediction_files = sorted(predictions_dir.glob("predictions_*.parquet"))
    
    if not prediction_files:
        print("No prediction files found, skipping distribution plot")
        return
    
    # Sample up to 10 time periods evenly
    if len(prediction_files) > 10:
        indices = np.linspace(0, len(prediction_files) - 1, 10, dtype=int)
        prediction_files = [prediction_files[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for pred_file in prediction_files:
        df = pd.read_parquet(pred_file)
        # Extract snapshot date portion from filename tail (YYYY_MM_DD)
        stem = pred_file.stem
        # filenames like predictions_<MODEL>_YYYY_MM_DD
        parts = stem.split('_')
        # Robust: last 3 parts are date tokens
        date_token = '_'.join(parts[-3:])
        # Parse underscore format explicitly to avoid parser errors
        try:
            date_dt = datetime.strptime(date_token, '%Y_%m_%d')
        except Exception:
            # Fallback: try replacing underscores with dashes
            try:
                date_dt = pd.to_datetime(date_token.replace('_', '-'))
            except Exception:
                print(f"Warning: could not parse date from filename {pred_file.name}; skipping")
                continue
        # Filter by configured monitoring period
        start_dt = pd.to_datetime(MONITOR_START_DATE)
        end_dt = pd.to_datetime(MONITOR_END_DATE)
        if not (start_dt <= date_dt <= end_dt):
            continue
        date_str = date_dt.strftime('%Y-%m-%d')

        # Normalize to 'prediction' column for plotting
        if 'prediction' not in df.columns:
            if 'prediction_proba' in df.columns:
                df = df.rename(columns={'prediction_proba': 'prediction'})
            else:
                print(f"Warning: file {pred_file.name} has no 'prediction' or 'prediction_proba' column; skipping")
                continue

        # CUSTOM: KDE plot with filled area instead of histogram
        try:
            from scipy.stats import gaussian_kde
            density = gaussian_kde(df['prediction'])
            xs = np.linspace(0, 1, 200)
            ys = density(xs)
            ax.plot(xs, ys, linewidth=2.5, label=date_str, alpha=0.8)
            ax.fill_between(xs, ys, alpha=0.15)
        except:
            # Fallback to histogram if KDE fails
            ax.hist(df['prediction'], bins=50, alpha=0.5, label=date_str, density=True)
    
    # CUSTOM: Enhanced labels and styling
    ax.set_xlabel('Predicted Default Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('📉 Prediction Probability Distribution Evolution', fontsize=14, 
                fontweight='bold', color=CUSTOM_COLORS['primary'])
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, title='Date', title_fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    output_file = output_dir / 'prediction_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def generate_monitoring_summary_report(df, output_dir):
    """
    Generate a text summary report
    
    Args:
        df: DataFrame with monitoring history
        output_dir: Directory to save report
    """
    report = []
    report.append("="*70)
    report.append("MODEL MONITORING SUMMARY REPORT")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nMonitoring Period: {df['snapshot_date'].min().strftime('%Y-%m-%d')} to {df['snapshot_date'].max().strftime('%Y-%m-%d')}")
    report.append(f"Number of Monitoring Points: {len(df)}")
    report.append(f"Model: {df['model_name'].iloc[0]}")
    
    report.append("\n" + "="*70)
    report.append("LATEST METRICS")
    report.append("="*70)
    latest = df.iloc[-1]
    report.append(f"\nSnapshot Date: {latest['snapshot_date'].strftime('%Y-%m-%d')}")
    report.append(f"AUC-ROC:     {latest['auc_roc']:.4f}")
    report.append(f"Accuracy:    {latest['accuracy']:.4f}")
    report.append(f"Precision:   {latest['precision']:.4f}")
    report.append(f"Recall:      {latest['recall']:.4f}")
    report.append(f"F1-Score:    {latest['f1_score']:.4f}")
    if 'psi' in latest:
        report.append(f"PSI:         {latest['psi']:.4f}")
    
    report.append("\n" + "="*70)
    report.append("AVERAGE METRICS")
    report.append("="*70)
    report.append(f"\nAUC-ROC:     {df['auc_roc'].mean():.4f}")
    report.append(f"Accuracy:    {df['accuracy'].mean():.4f}")
    report.append(f"Precision:   {df['precision'].mean():.4f}")
    report.append(f"Recall:      {df['recall'].mean():.4f}")
    report.append(f"F1-Score:    {df['f1_score'].mean():.4f}")
    if 'psi' in df.columns:
        report.append(f"PSI:         {df['psi'].mean():.4f}")
    
    report.append("\n" + "="*70)
    report.append("THRESHOLD COMPLIANCE")
    report.append("="*70)
    if 'check_all_passed' in df.columns:
        compliance_rate = df['check_all_passed'].mean() * 100
        report.append(f"\nOverall Compliance Rate: {compliance_rate:.1f}%")
    
    report.append("\n" + "="*70)
    
    # Save report
    output_file = output_dir / 'monitoring_summary_report.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Saved: {output_file}")
    
    # Also print to console
    print('\n'.join(report))


def main():
    """Main visualization pipeline"""
    print(f"\n{'='*70}")
    print(f"Model Monitoring Visualization")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = RESULTS_PATH / "monitoring_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load monitoring history
    df = load_monitoring_history()
    
    if df is None or len(df) < 2:
        print("Insufficient monitoring data for visualization (need at least 2 records)")
        return
    
    print("Generating visualizations...\n")
    
    # Generate plots
    plot_performance_metrics(df, output_dir)
    plot_psi_over_time(df, output_dir)
    plot_confusion_matrix_trend(df, output_dir)
    plot_threshold_compliance(df, output_dir)
    plot_prediction_distribution(output_dir)
    
    # Generate summary report
    generate_monitoring_summary_report(df, output_dir)
    
    print(f"\n{'='*70}")
    print(f"Visualization completed successfully")
    print(f"All charts saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
