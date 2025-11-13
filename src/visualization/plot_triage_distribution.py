"""Visualization utilities for triage data."""

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import text

from src.database.db_utils import get_engine


def plot_triage_distribution():
    """
    Plot distribution of triage labels.
    
    Returns:
        matplotlib figure
    """
    engine = get_engine()
    
    query = text("""
        SELECT triage_label, COUNT(*) as count
        FROM triage_predictions
        GROUP BY triage_label
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No triage data available", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(df["triage_label"], df["count"])
    ax.set_xlabel("Triage Label")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Triage Labels")
    plt.tight_layout()
    
    return fig


def plot_severity_vs_risk():
    """
    Plot severity vs risk score scatter chart.
    
    Returns:
        matplotlib figure
    """
    engine = get_engine()
    
    query = text("""
        SELECT sr.parsed_severity, tp.risk_score, tp.triage_label
        FROM symptom_reports sr
        JOIN triage_predictions tp ON sr.id = tp.symptom_report_id
        WHERE sr.parsed_severity IS NOT NULL
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for label in df["triage_label"].unique():
        subset = df[df["triage_label"] == label]
        ax.scatter(subset["parsed_severity"], subset["risk_score"], label=label, alpha=0.6)
    
    ax.set_xlabel("Symptom Severity")
    ax.set_ylabel("Risk Score")
    ax.set_title("Symptom Severity vs Risk Score")
    ax.legend()
    plt.tight_layout()
    
    return fig

