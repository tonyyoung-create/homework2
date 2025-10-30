"""
Visualization utilities for spam detection analysis.
"""
import logging
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go

# Optional dependency: wordcloud. Provide a graceful fallback when unavailable.
try:
    from wordcloud import WordCloud  # type: ignore
    _HAS_WORDCLOUD = True
except Exception:  # pragma: no cover - environment dependent
    WordCloud = None
    _HAS_WORDCLOUD = False
    logging.getLogger(__name__).warning(
        "wordcloud package not available; falling back to bar-chart summary"
    )
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_wordcloud(text, title='Word Cloud'):
    """
    Generate word cloud visualization.
    
    Args:
        text (str): Text to visualize
        title (str): Plot title
    """
    if _HAS_WORDCLOUD and WordCloud is not None:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
            text
        )
        # WordCloud exposes an image array; use px.imshow to display it
        fig = px.imshow(wordcloud.to_array())
        fig.update_layout(title=title)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig

    # Fallback: show top N words as a bar chart
    words = [w.strip() for w in str(text).lower().split() if w.strip()]
    top_n = 30
    counts = Counter(words).most_common(top_n)
    if not counts:
        # Empty input: return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no words to display)")
        return fig

    df = px.data.tips()  # small dummy to satisfy px API when empty — will be replaced
    # Build a DataFrame-like structure for px.bar
    words_list, counts_list = zip(*counts)
    import pandas as pd

    df = pd.DataFrame({"word": words_list, "count": counts_list})
    fig = px.bar(df, x="word", y="count", title=f"{title} (fallback - top words)")
    fig.update_layout(xaxis_tickangle=-45, margin=dict(b=140))
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance bars.
    
    Args:
        model: Trained model with feature_importances_
        feature_names (list): List of feature names
    """
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(
        importances,
        x='feature',
        y='importance',
        title='Feature Importance'
    )
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        labels=dict(
            x="Predicted Label",
            y="True Label"
        ),
        x=['Not Spam', 'Spam'],
        y=['Not Spam', 'Spam'],
        title='Confusion Matrix'
    )
    return fig

def plot_roc_curve(y_true, y_prob):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            name=f'ROC curve (AUC = {roc_auc:.2f})'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line=dict(dash='dash'),
            name='Random'
        )
    )
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    return fig