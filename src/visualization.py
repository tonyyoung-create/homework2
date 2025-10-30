"""
Visualization utilities for spam detection analysis.
"""
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
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
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig = px.imshow(wordcloud)
    fig.update_layout(title=title)
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