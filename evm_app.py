"""
EVM (Earned Value Management) Calculator - Educational Application
Based on IPMA (International Project Management Association) standards
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import base64

# Page configuration
st.set_page_config(
    page_title="EVM Calculator - IPMA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (minimal - theme compatible)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== EVM CALCULATION FUNCTIONS ====================

def calculate_evm_metrics(pv: float, ev: float, ac: float, bac: float) -> dict:
    """Calculate all EVM metrics"""
    metrics = {}

    # Basic metrics
    metrics['PV'] = pv
    metrics['EV'] = ev
    metrics['AC'] = ac
    metrics['BAC'] = bac

    # Variances
    metrics['SV'] = ev - pv  # Schedule Variance
    metrics['CV'] = ev - ac  # Cost Variance

    # Performance Indices
    metrics['SPI'] = ev / pv if pv != 0 else 0  # Schedule Performance Index
    metrics['CPI'] = ev / ac if ac != 0 else 0  # Cost Performance Index

    # Percent Complete
    metrics['PC_planned'] = (pv / bac * 100) if bac != 0 else 0
    metrics['PC_earned'] = (ev / bac * 100) if bac != 0 else 0

    # Estimates at Completion
    metrics['EAC_typical'] = bac / metrics['CPI'] if metrics['CPI'] != 0 else 0  # Typical variance
    metrics['EAC_atypical'] = ac + (bac - ev)  # Atypical variance
    metrics['EAC_combined'] = ac + (bac - ev) / (metrics['CPI'] * metrics['SPI']) if (metrics['CPI'] * metrics['SPI']) != 0 else 0

    # Estimate to Complete
    metrics['ETC_typical'] = metrics['EAC_typical'] - ac
    metrics['ETC_atypical'] = bac - ev

    # Variance at Completion
    metrics['VAC'] = bac - metrics['EAC_typical']

    # To-Complete Performance Index
    metrics['TCPI_BAC'] = (bac - ev) / (bac - ac) if (bac - ac) != 0 else 0
    metrics['TCPI_EAC'] = (bac - ev) / (metrics['EAC_typical'] - ac) if (metrics['EAC_typical'] - ac) != 0 else 0

    return metrics


def get_status_color(value: float, threshold_good: float = 1.0, higher_is_better: bool = True) -> str:
    """Return color based on metric value"""
    if higher_is_better:
        if value >= threshold_good:
            return "green"
        elif value >= threshold_good * 0.9:
            return "orange"
        else:
            return "red"
    else:
        if value <= threshold_good:
            return "green"
        elif value <= threshold_good * 1.1:
            return "orange"
        else:
            return "red"


def interpret_metric(metric_name: str, value: float) -> tuple:
    """Return interpretation text and status for a metric"""
    interpretations = {
        'SV': {
            'positive': ("✅ **Ahead of Schedule**: The project has earned more value than planned at this point in time.", "good"),
            'negative': ("⚠️ **Behind Schedule**: The project has earned less value than planned. Schedule recovery actions may be needed.", "bad"),
            'zero': ("➡️ **On Schedule**: The project is exactly on track with the planned schedule.", "neutral")
        },
        'CV': {
            'positive': ("✅ **Under Budget**: The project is spending less than the value of work completed.", "good"),
            'negative': ("⚠️ **Over Budget**: The project is spending more than the value of work completed. Cost control measures may be needed.", "bad"),
            'zero': ("➡️ **On Budget**: Actual costs exactly match the earned value.", "neutral")
        },
        'SPI': {
            'positive': ("✅ **Efficient Schedule Performance**: For every unit of work planned, more than one unit is being accomplished.", "good"),
            'negative': ("⚠️ **Inefficient Schedule Performance**: The project is only accomplishing {:.0%} of the planned work rate.".format(value), "bad"),
            'zero': ("➡️ **On Track**: Schedule performance is exactly as planned.", "neutral")
        },
        'CPI': {
            'positive': ("✅ **Efficient Cost Performance**: For every dollar spent, more than one dollar of value is being earned.", "good"),
            'negative': ("⚠️ **Inefficient Cost Performance**: For every unit spent, only {:.2f} of value is being earned.".format(value), "bad"),
            'zero': ("➡️ **On Budget**: Cost performance is exactly as planned.", "neutral")
        },
        'VAC': {
            'positive': ("✅ **Expected Savings**: The project is forecast to complete under the original budget.", "good"),
            'negative': ("⚠️ **Expected Overrun**: The project is forecast to exceed the original budget by this amount.", "bad"),
            'zero': ("➡️ **On Target**: The project is expected to complete exactly at budget.", "neutral")
        },
        'TCPI': {
            'positive': ("✅ **Achievable Target**: The required future efficiency ({:.2f}) is achievable.".format(value), "good"),
            'negative': ("⚠️ **Challenging Target**: A CPI of {:.2f} is required for remaining work - this may be difficult to achieve.".format(value), "bad"),
            'zero': ("➡️ **Neutral**: Future performance needs to match planned efficiency.", "neutral")
        }
    }

    if metric_name in ['SV', 'CV', 'VAC']:
        if value > 0:
            return interpretations[metric_name]['positive']
        elif value < 0:
            return interpretations[metric_name]['negative']
        else:
            return interpretations[metric_name]['zero']
    elif metric_name in ['SPI', 'CPI']:
        if value > 1:
            return interpretations[metric_name]['positive']
        elif value < 1:
            return interpretations[metric_name]['negative']
        else:
            return interpretations[metric_name]['zero']
    elif metric_name == 'TCPI':
        if value <= 1.1:
            return interpretations[metric_name]['positive']
        else:
            return interpretations[metric_name]['negative']

    return ("No interpretation available.", "neutral")


# ==================== VISUALIZATION FUNCTIONS ====================

def create_s_curve(periods_data: pd.DataFrame) -> go.Figure:
    """Create S-Curve chart showing PV, EV, and AC over time"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=periods_data['Period'],
        y=periods_data['PV_cumulative'],
        mode='lines+markers',
        name='Planned Value (PV)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=periods_data['Period'],
        y=periods_data['EV_cumulative'],
        mode='lines+markers',
        name='Earned Value (EV)',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=periods_data['Period'],
        y=periods_data['AC_cumulative'],
        mode='lines+markers',
        name='Actual Cost (AC)',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='S-Curve: Cumulative Project Performance',
        xaxis_title='Time Period',
        yaxis_title='Cost',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified',
        height=500
    )

    return fig


def create_variance_chart(periods_data: pd.DataFrame) -> go.Figure:
    """Create variance chart showing SV and CV over time"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Schedule Variance (SV)', 'Cost Variance (CV)'),
        vertical_spacing=0.15
    )

    colors_sv = ['green' if v >= 0 else 'red' for v in periods_data['SV']]
    colors_cv = ['green' if v >= 0 else 'red' for v in periods_data['CV']]

    fig.add_trace(
        go.Bar(x=periods_data['Period'], y=periods_data['SV'],
               marker_color=colors_sv, name='SV'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=periods_data['Period'], y=periods_data['CV'],
               marker_color=colors_cv, name='CV'),
        row=2, col=1
    )

    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(title_text="Time Period", row=2, col=1)
    fig.update_yaxes(title_text="Variance", row=1, col=1)
    fig.update_yaxes(title_text="Variance", row=2, col=1)

    return fig


def create_index_chart(periods_data: pd.DataFrame) -> go.Figure:
    """Create performance index chart showing SPI and CPI over time"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=periods_data['Period'],
        y=periods_data['SPI'],
        mode='lines+markers',
        name='Schedule Performance Index (SPI)',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=periods_data['Period'],
        y=periods_data['CPI'],
        mode='lines+markers',
        name='Cost Performance Index (CPI)',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=8)
    ))

    # Add reference line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Target (1.0)")

    # Add shaded regions
    fig.add_hrect(y0=1.0, y1=max(periods_data['SPI'].max(), periods_data['CPI'].max(), 1.5),
                  fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=min(periods_data['SPI'].min(), periods_data['CPI'].min(), 0.5), y1=1.0,
                  fillcolor="red", opacity=0.1, line_width=0)

    fig.update_layout(
        title='Performance Indices Over Time',
        xaxis_title='Time Period',
        yaxis_title='Index Value',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified',
        height=400
    )

    return fig


def create_forecast_chart(bac: float, eac: float, ac: float, ev: float) -> go.Figure:
    """Create forecast comparison chart"""
    categories = ['Budget at Completion (BAC)', 'Estimate at Completion (EAC)',
                  'Actual Cost (AC)', 'Earned Value (EV)']
    values = [bac, eac, ac, ev]
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

    # Calculate max value for y-axis with padding for text labels
    max_value = max(values)
    y_max = max_value * 1.15  # 15% padding for text labels

    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors, text=[f'{v:,.0f}' for v in values],
               textposition='outside')
    ])

    fig.update_layout(
        title='Budget vs. Forecast Comparison',
        yaxis_title='Cost',
        yaxis=dict(range=[0, y_max], tickformat=',.0f'),
        height=400,
        showlegend=False
    )

    return fig


def create_evm_performance_graph(
    planned_value_data: list,
    actual_cost_data: list,
    earned_value_data: list,
    time_unit: str = "Period"
) -> go.Figure:
    """
    Create EVM Performance Graph visualizing PV, AC, and EV over time.

    Args:
        planned_value_data: Cumulative Planned Value data points
        actual_cost_data: Cumulative Actual Cost data points
        earned_value_data: Cumulative Earned Value data points
        time_unit: Label for time axis (e.g., "Week", "Month", "Day")

    Returns:
        Plotly Figure object
    """
    # Create time periods (1-indexed)
    time_periods = list(range(1, len(planned_value_data) + 1))

    fig = go.Figure()

    # Planned Value (PV) - Blue dashed line
    fig.add_trace(go.Scatter(
        x=time_periods,
        y=planned_value_data,
        mode='lines+markers',
        name='Planned Value (PV)',
        line=dict(color='#1f77b4', width=3, dash='dash'),
        marker=dict(size=10, symbol='circle'),
        hovertemplate='Period %{x}<br>PV: %{y:,.0f}<extra></extra>'
    ))

    # Actual Cost (AC) - Red solid line
    fig.add_trace(go.Scatter(
        x=time_periods,
        y=actual_cost_data,
        mode='lines+markers',
        name='Actual Cost (AC)',
        line=dict(color='#d62728', width=3),
        marker=dict(size=10, symbol='square'),
        hovertemplate='Period %{x}<br>AC: %{y:,.0f}<extra></extra>'
    ))

    # Earned Value (EV) - Green solid line
    fig.add_trace(go.Scatter(
        x=time_periods,
        y=earned_value_data,
        mode='lines+markers',
        name='Earned Value (EV)',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='Period %{x}<br>EV: %{y:,.0f}<extra></extra>'
    ))

    # Calculate max value for y-axis range
    max_value = max(
        max(planned_value_data) if planned_value_data else 0,
        max(actual_cost_data) if actual_cost_data else 0,
        max(earned_value_data) if earned_value_data else 0
    )

    fig.update_layout(
        title={
            'text': 'EVM Performance Graph',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20, color='#333')
        },
        xaxis_title=f'Time ({time_unit})',
        yaxis_title='Cost',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#ccc',
            borderwidth=1
        ),
        hovermode='x unified',
        height=500,
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#e0e0e0',
            dtick=1,
            tick0=1
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#e0e0e0',
            range=[0, max_value * 1.1],
            tickformat=',.0f'
        ),
        font=dict(family="Arial, sans-serif", size=12)
    )

    return fig


def create_classic_evm_chart(
    planned_value_data: list,
    actual_cost_data: list,
    earned_value_data: list,
    current_period: int = None,
    bac: float = None,
    eac: float = None
) -> go.Figure:
    """
    Create classic EVM S-Curve chart with Cost Variance, Schedule Variance,
    EAC forecast, and Management Reserve - following PMBOK/IPMA standards.

    Args:
        planned_value_data: Cumulative Planned Value data points
        actual_cost_data: Cumulative Actual Cost data points
        earned_value_data: Cumulative Earned Value data points
        current_period: Current reporting period (Time Now)
        bac: Budget at Completion
        eac: Estimate at Completion
        currency: Currency symbol

    Returns:
        Plotly Figure object
    """
    n_periods = len(planned_value_data)
    time_periods = list(range(1, n_periods + 1))

    # Determine current period (last period with actual data)
    if current_period is None:
        for i in range(len(earned_value_data) - 1, -1, -1):
            if earned_value_data[i] > 0 or actual_cost_data[i] > 0:
                current_period = i + 1
                break
        if current_period is None:
            current_period = n_periods

    # Get values at current period
    idx = min(current_period - 1, len(planned_value_data) - 1)
    pv_current = planned_value_data[idx]
    ev_current = earned_value_data[idx] if idx < len(earned_value_data) else 0
    ac_current = actual_cost_data[idx] if idx < len(actual_cost_data) else 0

    # Calculate BAC if not provided
    if bac is None:
        bac = max(planned_value_data)

    # Calculate performance indices
    cpi = ev_current / ac_current if ac_current > 0 else 1
    spi = ev_current / pv_current if pv_current > 0 else 1

    # Calculate EAC if not provided (using typical formula: BAC / CPI)
    if eac is None:
        eac = bac / cpi if cpi > 0 else bac

    # Calculate variances
    cv = ev_current - ac_current  # Cost Variance
    sv = ev_current - pv_current  # Schedule Variance

    # Calculate SV-time (schedule variance in time units)
    # Find when PV reached current EV level
    sv_time = 0
    if ev_current > 0:
        for i, pv_val in enumerate(planned_value_data):
            if pv_val >= ev_current:
                sv_time = current_period - (i + 1)
                break

    # Estimate completion period based on SPI
    completion_period = n_periods
    if spi > 0 and spi != 1:
        remaining_periods = (n_periods - current_period) / spi
        completion_period = current_period + remaining_periods

    fig = go.Figure()

    # === 1. PLANNED VALUE (PV) - Blue S-curve (PMB - Performance Measurement Baseline) ===
    fig.add_trace(go.Scatter(
        x=time_periods,
        y=planned_value_data,
        mode='lines',
        name='PV (BCWS)',
        line=dict(color='#2563EB', width=3),
        hovertemplate='Period %{x}<br>PV: %{y:,.0f}<extra>Planned Value</extra>'
    ))

    # === 2. EARNED VALUE (EV) - Green line ===
    ev_periods = list(range(1, current_period + 1))
    ev_values = earned_value_data[:current_period]
    fig.add_trace(go.Scatter(
        x=ev_periods,
        y=ev_values,
        mode='lines',
        name='EV (BCWP)',
        line=dict(color='#16A34A', width=3),
        hovertemplate='Period %{x}<br>EV: %{y:,.0f}<extra>Earned Value</extra>'
    ))

    # === 3. ACTUAL COST (AC) - Red line ===
    ac_periods = list(range(1, current_period + 1))
    ac_values = actual_cost_data[:current_period]
    fig.add_trace(go.Scatter(
        x=ac_periods,
        y=ac_values,
        mode='lines',
        name='AC (ACWP)',
        line=dict(color='#DC2626', width=3),
        hovertemplate='Period %{x}<br>AC: %{y:,.0f}<extra>Actual Cost</extra>'
    ))

    # === 4. EAC FORECAST LINE - Cyan dashed line ===
    # Project from current AC to EAC at completion
    if eac > 0 and completion_period > current_period:
        eac_x = [current_period, completion_period]
        eac_y = [ac_current, eac]
        fig.add_trace(go.Scatter(
            x=eac_x,
            y=eac_y,
            mode='lines',
            name='EAC',
            line=dict(color='#06B6D4', width=2, dash='dash'),
            hovertemplate='Period %{x}<br>Forecast: %{y:,.0f}<extra>EAC Projection</extra>'
        ))

    # === 5. MANAGEMENT RESERVE AREA ===
    if eac > bac:
        # Calculate shortage (negative VAC means over budget)
        shortage = eac - bac  # Amount over budget

        # Show area between BAC and EAC as Management Reserve needed
        fig.add_hrect(
            y0=bac, y1=eac,
            fillcolor="rgba(251, 191, 36, 0.3)",
            line_width=0
        )
        # Separate annotation for Management Reserve with better visibility
        fig.add_annotation(
            x=0.5,
            y=(bac + eac) / 2,
            text=f"<b>Management Reserve</b><br>Shortfall: {shortage:,.0f}",
            showarrow=False,
            font=dict(size=11, color='#B45309'),
            bgcolor='rgba(251, 191, 36, 0.8)',
            bordercolor='#B45309',
            borderwidth=1,
            borderpad=4
        )

    # === 6. BAC LINE ===
    fig.add_hline(
        y=bac,
        line=dict(color='#1E3A8A', width=2, dash='dash'),
    )
    fig.add_annotation(
        x=n_periods + 0.5,
        y=bac,
        text=f"<b>BAC</b><br>{bac:,.0f}",
        showarrow=False,
        xanchor='left',
        font=dict(size=11, color='#1E3A8A')
    )

    # === 7. EAC LINE ===
    if eac != bac:
        fig.add_hline(
            y=eac,
            line=dict(color='#06B6D4', width=1, dash='dot'),
        )
        fig.add_annotation(
            x=n_periods + 0.5,
            y=eac,
            text=f"<b>EAC</b><br>{eac:,.0f}",
            showarrow=False,
            xanchor='left',
            font=dict(size=11, color='#06B6D4')
        )

    # === 8. TIME NOW (Status Date) vertical line ===
    fig.add_vline(
        x=current_period,
        line=dict(color='#6B7280', width=2, dash='dash'),
    )
    fig.add_annotation(
        x=current_period,
        y=0,
        text="<b>Time Now</b>",
        showarrow=False,
        yshift=-25,
        font=dict(size=11, color='#6B7280')
    )

    # === 9. COMPLETION DATE vertical line ===
    if completion_period > n_periods:
        fig.add_vline(
            x=completion_period,
            line=dict(color='#1E3A8A', width=2, dash='dash'),
        )
        fig.add_annotation(
            x=completion_period,
            y=0,
            text="<b>Completion<br>Date</b>",
            showarrow=False,
            yshift=-25,
            font=dict(size=10, color='#1E3A8A')
        )

    # === 10. COST VARIANCE (CV) annotation ===
    if ev_current > 0 and ac_current > 0 and abs(cv) > 0.01:
        # Dimension line for CV - positioned to the left of Time Now line
        cv_x = current_period - 0.6

        # Vertical dimension line
        fig.add_trace(go.Scatter(
            x=[cv_x, cv_x],
            y=[ev_current, ac_current],
            mode='lines',
            line=dict(color='#DC2626', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Horizontal ticks at ends (dimension line style)
        tick_width = 0.15
        fig.add_trace(go.Scatter(
            x=[cv_x - tick_width, cv_x + tick_width],
            y=[ev_current, ev_current],
            mode='lines',
            line=dict(color='#DC2626', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[cv_x - tick_width, cv_x + tick_width],
            y=[ac_current, ac_current],
            mode='lines',
            line=dict(color='#DC2626', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Extension lines from points to dimension line
        fig.add_trace(go.Scatter(
            x=[current_period, cv_x + tick_width],
            y=[ev_current, ev_current],
            mode='lines',
            line=dict(color='#DC2626', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[current_period, cv_x + tick_width],
            y=[ac_current, ac_current],
            mode='lines',
            line=dict(color='#DC2626', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

        cv_mid = (ev_current + ac_current) / 2
        fig.add_annotation(
            x=cv_x - 0.2,
            y=cv_mid,
            text=f"<b>Cost Variance (CV)</b><br>{cv:+,.0f}",
            showarrow=False,
            xanchor='right',
            font=dict(size=10, color='#DC2626'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#DC2626',
            borderwidth=1,
            borderpad=4
        )

    # === 11. SCHEDULE VARIANCE (SV) annotation ===
    if ev_current > 0 and pv_current > 0 and abs(sv) > 0.01:
        # Dimension line for SV - positioned to the right of Time Now line
        sv_x = current_period + 0.6

        # Vertical dimension line
        fig.add_trace(go.Scatter(
            x=[sv_x, sv_x],
            y=[ev_current, pv_current],
            mode='lines',
            line=dict(color='#1E3A8A', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Horizontal ticks at ends (dimension line style)
        tick_width = 0.15
        fig.add_trace(go.Scatter(
            x=[sv_x - tick_width, sv_x + tick_width],
            y=[ev_current, ev_current],
            mode='lines',
            line=dict(color='#1E3A8A', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[sv_x - tick_width, sv_x + tick_width],
            y=[pv_current, pv_current],
            mode='lines',
            line=dict(color='#1E3A8A', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Extension lines from Time Now to dimension line
        fig.add_trace(go.Scatter(
            x=[current_period, sv_x - tick_width],
            y=[ev_current, ev_current],
            mode='lines',
            line=dict(color='#1E3A8A', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[current_period, sv_x - tick_width],
            y=[pv_current, pv_current],
            mode='lines',
            line=dict(color='#1E3A8A', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

        sv_mid = (ev_current + pv_current) / 2
        fig.add_annotation(
            x=sv_x + 0.2,
            y=sv_mid,
            text=f"<b>Schedule Variance (SV)</b><br>{sv:+,.0f}",
            showarrow=False,
            xanchor='left',
            font=dict(size=10, color='#1E3A8A'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#1E3A8A',
            borderwidth=1,
            borderpad=4
        )

    # === 12. SV-time annotation ===
    if sv_time != 0 and ev_current > 0:
        # Find the period where PV equals current EV
        sv_time_period = current_period - sv_time
        if 0 < sv_time_period <= n_periods:
            fig.add_trace(go.Scatter(
                x=[sv_time_period, current_period],
                y=[ev_current, ev_current],
                mode='lines',
                line=dict(color='#9333EA', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_annotation(
                x=(sv_time_period + current_period) / 2,
                y=ev_current,
                text=f"<b>SV-time</b><br>{sv_time:+.1f} periods",
                showarrow=False,
                yshift=20,
                font=dict(size=10, color='#9333EA'),
                bgcolor='rgba(255,255,255,0.8)'
            )

    # === 13. CURVE LABELS ===
    # PV label
    pv_label_idx = min(n_periods - 2, int(n_periods * 0.7))
    fig.add_annotation(
        x=pv_label_idx + 1,
        y=planned_value_data[pv_label_idx] * 1.08,
        text="<b>PV (BCWS)</b>",
        showarrow=True,
        arrowhead=2,
        ax=-30,
        ay=-20,
        font=dict(size=10, color='#2563EB')
    )

    # AC label
    if len(ac_values) > 2:
        ac_label_idx = len(ac_values) // 2
        fig.add_annotation(
            x=ac_label_idx + 1,
            y=ac_values[ac_label_idx] * 1.1,
            text="<b>AC (ACWP)</b>",
            showarrow=True,
            arrowhead=2,
            ax=-40,
            ay=-20,
            font=dict(size=10, color='#DC2626')
        )

    # EV label
    if len(ev_values) > 2:
        ev_label_idx = len(ev_values) // 2
        fig.add_annotation(
            x=ev_label_idx + 1,
            y=ev_values[ev_label_idx] * 0.9,
            text="<b>EV (BCWP)</b>",
            showarrow=True,
            arrowhead=2,
            ax=-40,
            ay=20,
            font=dict(size=10, color='#16A34A')
        )

    # === LAYOUT ===
    max_y = max(bac, eac, max(planned_value_data), max(actual_cost_data) if actual_cost_data else 0) * 1.2
    max_x = max(n_periods, completion_period) + 1.5

    fig.update_layout(
        title={
            'text': '<b>Earned Value Management (EVM) Chart</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20)
        },
        xaxis_title='<b>Time</b>',
        yaxis_title='<b>Cost</b>',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=600,
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E7EB',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='#000000',
            range=[0, max_x],
            dtick=1
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E7EB',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='#000000',
            range=[0, max_y],
            tickformat=',.0f'
        ),
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(t=80, b=80, l=120, r=120)
    )

    return fig


def create_gauge_chart(value: float, title: str, min_val: float = 0, max_val: float = 2) -> go.Figure:
    """Create a gauge chart for performance indices"""
    if value >= 1:
        color = "green"
    elif value >= 0.9:
        color = "orange"
    else:
        color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.9], 'color': "#ffcccc"},
                {'range': [0.9, 1.0], 'color': "#ffffcc"},
                {'range': [1.0, 2.0], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 1.0
            }
        }
    ))

    fig.update_layout(height=250)
    return fig


# ==================== HTML REPORT GENERATION ====================

def generate_html_report(metrics: dict, periods_data: pd.DataFrame = None) -> str:
    """Generate a comprehensive HTML report for EVM analysis"""

    # Determine project status
    spi = metrics['SPI']
    cpi = metrics['CPI']

    if spi >= 1 and cpi >= 1:
        status = "EXCELLENT - Project is ahead of schedule AND under budget"
        status_color = "#28a745"
    elif spi >= 1 and cpi < 1:
        status = "ATTENTION - Ahead of schedule but over budget"
        status_color = "#ffc107"
    elif spi < 1 and cpi >= 1:
        status = "ATTENTION - Under budget but behind schedule"
        status_color = "#ffc107"
    else:
        status = "CRITICAL - Behind schedule AND over budget"
        status_color = "#dc3545"

    # Build recommendations
    recommendations = []
    if metrics['SPI'] < 1:
        recommendations.append("Schedule Recovery: Consider adding resources, working overtime, or fast-tracking activities.")
    if metrics['CPI'] < 1:
        recommendations.append("Cost Control: Review and reduce non-essential expenses, negotiate with vendors.")
    if metrics['TCPI_BAC'] > 1.2:
        recommendations.append("Re-baseline: The required future performance may be unrealistic.")
    if metrics['SPI'] >= 1 and metrics['CPI'] >= 1:
        recommendations.append("Maintain Performance: Continue current management practices.")
    if metrics['VAC'] < 0:
        recommendations.append("Request Additional Funding: Prepare a change request for additional budget.")

    recommendations_html = "".join([f"<li>{r}</li>" for r in recommendations]) if recommendations else "<li>Project is performing well.</li>"

    # Build periods table if available
    periods_table_html = ""
    if periods_data is not None and len(periods_data) > 0:
        periods_rows = ""
        for _, row in periods_data.iterrows():
            periods_rows += f"""
            <tr>
                <td>{int(row['Period'])}</td>
                <td>{row['PV_cumulative']:,.0f}</td>
                <td>{row['EV_cumulative']:,.0f}</td>
                <td>{row['AC_cumulative']:,.0f}</td>
                <td>{row['SV']:+,.0f}</td>
                <td>{row['CV']:+,.0f}</td>
                <td>{row['SPI']:.2f}</td>
                <td>{row['CPI']:.2f}</td>
            </tr>
            """
        periods_table_html = f"""
        <h2>4. Period-by-Period Data</h2>
        <table>
            <tr>
                <th>Period</th><th>PV</th><th>EV</th><th>AC</th>
                <th>SV</th><th>CV</th><th>SPI</th><th>CPI</th>
            </tr>
            {periods_rows}
        </table>
        """

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EVM Report - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .report {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a365d;
            border-bottom: 3px solid #3182ce;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #2d3748;
            background: #edf2f7;
            padding: 10px 15px;
            margin: 30px 0 15px 0;
            border-left: 4px solid #3182ce;
        }}
        .meta {{
            color: #718096;
            margin-bottom: 20px;
        }}
        .status {{
            padding: 15px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 1.1em;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #3182ce;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #f7fafc;
        }}
        tr:hover {{
            background: #edf2f7;
        }}
        .metric-value {{
            font-weight: bold;
            font-family: 'Consolas', monospace;
        }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        ul {{
            margin: 15px 0;
            padding-left: 25px;
        }}
        li {{
            margin: 8px 0;
        }}
        .formula {{
            font-family: 'Consolas', monospace;
            background: #f7fafc;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #718096;
            font-size: 0.9em;
        }}
        @media print {{
            body {{ background: white; }}
            .report {{ box-shadow: none; }}
            h2 {{ page-break-after: avoid; }}
            table {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="report">
        <h1>EVM Report - Earned Value Management Analysis</h1>
        <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <h2>1. Executive Summary</h2>
        <div class="status" style="background: {status_color}20; border-left: 4px solid {status_color}; color: {status_color};">
            Project Status: {status}
        </div>

        <h2>2. Key Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
            <tr><td>Budget at Completion (BAC)</td><td class="metric-value">{metrics['BAC']:,.0f}</td><td>Baseline</td></tr>
            <tr><td>Planned Value (PV)</td><td class="metric-value">{metrics['PV']:,.0f}</td><td>{metrics['PC_planned']:.1f}% planned</td></tr>
            <tr><td>Earned Value (EV)</td><td class="metric-value">{metrics['EV']:,.0f}</td><td>{metrics['PC_earned']:.1f}% earned</td></tr>
            <tr><td>Actual Cost (AC)</td><td class="metric-value">{metrics['AC']:,.0f}</td><td>Spent to date</td></tr>
            <tr><td>Schedule Variance (SV)</td><td class="metric-value {'positive' if metrics['SV'] >= 0 else 'negative'}">{metrics['SV']:+,.0f}</td><td>{'Ahead' if metrics['SV'] >= 0 else 'Behind'} of schedule</td></tr>
            <tr><td>Cost Variance (CV)</td><td class="metric-value {'positive' if metrics['CV'] >= 0 else 'negative'}">{metrics['CV']:+,.0f}</td><td>{'Under' if metrics['CV'] >= 0 else 'Over'} budget</td></tr>
            <tr><td>Schedule Performance Index (SPI)</td><td class="metric-value {'positive' if metrics['SPI'] >= 1 else 'negative'}">{metrics['SPI']:.3f}</td><td>{'On/Ahead' if metrics['SPI'] >= 1 else 'Behind'} schedule</td></tr>
            <tr><td>Cost Performance Index (CPI)</td><td class="metric-value {'positive' if metrics['CPI'] >= 1 else 'negative'}">{metrics['CPI']:.3f}</td><td>{'On/Under' if metrics['CPI'] >= 1 else 'Over'} budget</td></tr>
        </table>

        <h2>3. Forecast Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
            <tr><td>Estimate at Completion (EAC)</td><td class="metric-value">{metrics['EAC_typical']:,.0f}</td><td>Forecasted total cost</td></tr>
            <tr><td>Estimate to Complete (ETC)</td><td class="metric-value">{metrics['ETC_typical']:,.0f}</td><td>Remaining cost to finish</td></tr>
            <tr><td>Variance at Completion (VAC)</td><td class="metric-value {'positive' if metrics['VAC'] >= 0 else 'negative'}">{metrics['VAC']:+,.0f}</td><td>{'Savings' if metrics['VAC'] >= 0 else 'Overrun'} expected</td></tr>
            <tr><td>To-Complete Performance Index (TCPI)</td><td class="metric-value">{metrics['TCPI_BAC']:.3f}</td><td>Required CPI for remaining work</td></tr>
        </table>

        {periods_table_html}

        <h2>{'5' if periods_data is not None and len(periods_data) > 0 else '4'}. Recommendations</h2>
        <ul>
            {recommendations_html}
        </ul>

        <h2>{'6' if periods_data is not None and len(periods_data) > 0 else '5'}. EVM Formulas Reference</h2>
        <table>
            <tr><th>Metric</th><th>Formula</th><th>Interpretation</th></tr>
            <tr><td>Schedule Variance</td><td class="formula">SV = EV - PV</td><td>Positive = Ahead, Negative = Behind</td></tr>
            <tr><td>Cost Variance</td><td class="formula">CV = EV - AC</td><td>Positive = Under budget, Negative = Over</td></tr>
            <tr><td>Schedule Performance Index</td><td class="formula">SPI = EV / PV</td><td>&gt; 1 = Ahead, &lt; 1 = Behind</td></tr>
            <tr><td>Cost Performance Index</td><td class="formula">CPI = EV / AC</td><td>&gt; 1 = Under budget, &lt; 1 = Over</td></tr>
            <tr><td>Estimate at Completion</td><td class="formula">EAC = BAC / CPI</td><td>Forecasted total cost</td></tr>
            <tr><td>Estimate to Complete</td><td class="formula">ETC = EAC - AC</td><td>Cost to finish</td></tr>
            <tr><td>Variance at Completion</td><td class="formula">VAC = BAC - EAC</td><td>Expected variance at end</td></tr>
            <tr><td>To-Complete Performance Index</td><td class="formula">TCPI = (BAC-EV) / (BAC-AC)</td><td>Required CPI to meet budget</td></tr>
        </table>

        <div class="footer">
            <p>Generated by EVM Calculator - Based on IPMA Standards</p>
            <p>Developed by Boris Taliev</p>
        </div>
    </div>
</body>
</html>
    """

    return html


# ==================== MAIN APPLICATION ====================

def main():
    # Header
    st.title("📊 EVM Calculator")
    st.markdown("### Earned Value Management - Based on IPMA Standards")
    st.markdown("---")

    # Sidebar - Input Section
    with st.sidebar:
        st.header("📥 Project Data Input")

        input_mode = st.radio(
            "Input Mode:",
            ["Single Period Analysis", "Multi-Period Analysis", "Use Example Data"]
        )

        if input_mode == "Use Example Data":
            # Load example data
            st.info("Loading example project data...")
            bac = 100000
            periods = 10
            current_period = 6

            # Generate example data
            pv_per_period = [8000, 10000, 12000, 15000, 18000, 12000, 10000, 8000, 5000, 2000]
            ev_per_period = [7500, 9500, 11000, 13000, 15000, 10000, 0, 0, 0, 0]
            ac_per_period = [8500, 11000, 13000, 15000, 17000, 12000, 0, 0, 0, 0]

            periods_data = pd.DataFrame({
                'Period': range(1, periods + 1),
                'PV': pv_per_period,
                'EV': ev_per_period,
                'AC': ac_per_period
            })

            # Calculate cumulative values
            periods_data['PV_cumulative'] = periods_data['PV'].cumsum()
            periods_data['EV_cumulative'] = periods_data['EV'].cumsum()
            periods_data['AC_cumulative'] = periods_data['AC'].cumsum()

            # Calculate variances and indices for each period
            periods_data['SV'] = periods_data['EV_cumulative'] - periods_data['PV_cumulative']
            periods_data['CV'] = periods_data['EV_cumulative'] - periods_data['AC_cumulative']
            periods_data['SPI'] = periods_data['EV_cumulative'] / periods_data['PV_cumulative']
            periods_data['CPI'] = periods_data['EV_cumulative'] / periods_data['AC_cumulative']

            # Current values
            pv = periods_data.loc[current_period - 1, 'PV_cumulative']
            ev = periods_data.loc[current_period - 1, 'EV_cumulative']
            ac = periods_data.loc[current_period - 1, 'AC_cumulative']

            st.success(f"Example data loaded: {periods} periods, current period: {current_period}")

        elif input_mode == "Single Period Analysis":
            st.subheader("Enter Values")

            bac = st.number_input("Budget at Completion (BAC)",
                                  min_value=0.0, value=100000.0, step=1000.0,
                                  help="Total planned budget for the entire project")

            pv = st.number_input("Planned Value (PV)",
                                 min_value=0.0, value=50000.0, step=1000.0,
                                 help="Budgeted cost of work scheduled to date")

            ev = st.number_input("Earned Value (EV)",
                                 min_value=0.0, value=45000.0, step=1000.0,
                                 help="Budgeted cost of work actually completed")

            ac = st.number_input("Actual Cost (AC)",
                                 min_value=0.0, value=55000.0, step=1000.0,
                                 help="Actual cost incurred for work completed")

            periods_data = None

        else:  # Multi-Period Analysis
            st.subheader("Project Setup")

            bac = st.number_input("Budget at Completion (BAC)",
                                  min_value=0.0, value=100000.0, step=1000.0)

            periods = st.number_input("Number of Periods",
                                      min_value=2, max_value=24, value=6, step=1)

            st.subheader("Enter Period Data (Cumulative Values)")
            st.caption("Enter cumulative (running total) values for each period, not per-period increments")

            # Create input dataframe
            periods_data = pd.DataFrame({
                'Period': range(1, periods + 1),
                'PV_cumulative': [0.0] * periods,
                'EV_cumulative': [0.0] * periods,
                'AC_cumulative': [0.0] * periods
            })

            edited_df = st.data_editor(
                periods_data,
                column_config={
                    "Period": st.column_config.NumberColumn("Period", disabled=True),
                    "PV_cumulative": st.column_config.NumberColumn("PV (cumulative)", min_value=0, format="%.0f"),
                    "EV_cumulative": st.column_config.NumberColumn("EV (cumulative)", min_value=0, format="%.0f"),
                    "AC_cumulative": st.column_config.NumberColumn("AC (cumulative)", min_value=0, format="%.0f"),
                },
                hide_index=True,
                use_container_width=True
            )

            periods_data = edited_df.copy()

            # Values are already cumulative - no need to sum
            # Calculate variances and indices
            periods_data['SV'] = periods_data['EV_cumulative'] - periods_data['PV_cumulative']
            periods_data['CV'] = periods_data['EV_cumulative'] - periods_data['AC_cumulative']
            periods_data['SPI'] = np.where(periods_data['PV_cumulative'] != 0,
                                           periods_data['EV_cumulative'] / periods_data['PV_cumulative'], 0)
            periods_data['CPI'] = np.where(periods_data['AC_cumulative'] != 0,
                                           periods_data['EV_cumulative'] / periods_data['AC_cumulative'], 0)

            # Get latest cumulative values
            pv = periods_data['PV_cumulative'].iloc[-1]
            ev = periods_data['EV_cumulative'].iloc[-1]
            ac = periods_data['AC_cumulative'].iloc[-1]

        st.markdown("---")
        calculate_button = st.button("🔄 Calculate EVM Metrics", type="primary", use_container_width=True)

    # Main Content Area
    if calculate_button or input_mode == "Use Example Data":
        # Calculate metrics
        metrics = calculate_evm_metrics(pv, ev, ac, bac)

        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Dashboard", "📐 Formulas & Calculations", "📊 Charts",
            "📝 Interpretations", "📚 EVM Theory"
        ])

        # ==================== TAB 1: DASHBOARD ====================
        with tab1:
            st.header("EVM Dashboard")

            # Key Metrics Row 1
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Budget at Completion (BAC)", f"{metrics['BAC']:,.0f}")
            with col2:
                st.metric("Planned Value (PV)", f"{metrics['PV']:,.0f}",
                         f"{metrics['PC_planned']:.1f}% complete")
            with col3:
                st.metric("Earned Value (EV)", f"{metrics['EV']:,.0f}",
                         f"{metrics['PC_earned']:.1f}% complete")
            with col4:
                st.metric("Actual Cost (AC)", f"{metrics['AC']:,.0f}")

            st.markdown("---")

            # Performance Indices with Gauges
            st.subheader("Performance Indices")
            col1, col2 = st.columns(2)

            with col1:
                fig_spi = create_gauge_chart(metrics['SPI'], "Schedule Performance Index (SPI)")
                st.plotly_chart(fig_spi, use_container_width=True)

            with col2:
                fig_cpi = create_gauge_chart(metrics['CPI'], "Cost Performance Index (CPI)")
                st.plotly_chart(fig_cpi, use_container_width=True)

            # Variances
            st.subheader("Variances")
            col1, col2, col3 = st.columns(3)

            with col1:
                sv_delta = "Ahead" if metrics['SV'] >= 0 else "Behind"
                st.metric("Schedule Variance (SV)", f"{metrics['SV']:,.0f}", sv_delta)

            with col2:
                cv_delta = "Under Budget" if metrics['CV'] >= 0 else "Over Budget"
                st.metric("Cost Variance (CV)", f"{metrics['CV']:,.0f}", cv_delta)

            with col3:
                vac_delta = "Savings" if metrics['VAC'] >= 0 else "Overrun"
                st.metric("Variance at Completion (VAC)", f"{metrics['VAC']:,.0f}", vac_delta)

            # Forecasts
            st.subheader("Forecasts")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Estimate at Completion (EAC)", f"{metrics['EAC_typical']:,.0f}",
                         f"{metrics['EAC_typical'] - metrics['BAC']:+,.0f} vs BAC")
            with col2:
                st.metric("Estimate to Complete (ETC)", f"{metrics['ETC_typical']:,.0f}")
            with col3:
                st.metric("To-Complete Performance Index", f"{metrics['TCPI_BAC']:.2f}",
                         "Required CPI for remaining work")

            # HTML Report Download Section
            st.markdown("---")
            st.subheader("📄 Download Report")

            # Generate HTML report
            html_report = generate_html_report(metrics=metrics, periods_data=periods_data)

            st.download_button(
                label="📥 Download HTML Report",
                data=html_report,
                file_name=f"EVM_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                type="secondary",
                use_container_width=True
            )
            st.caption("Open in browser and use Ctrl+P to print or save as PDF")

        # ==================== TAB 2: FORMULAS ====================
        with tab2:
            st.header("Formulas & Calculations")

            st.markdown("### Basic EVM Metrics")

            # PV, EV, AC
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Planned Value (PV)")
                st.latex(r"PV = \text{Budgeted Cost of Work Scheduled}")
                st.info(f"**Your Value:** PV = {metrics['PV']:,.0f}")

                st.markdown("#### Earned Value (EV)")
                st.latex(r"EV = \text{Budgeted Cost of Work Performed}")
                st.info(f"**Your Value:** EV = {metrics['EV']:,.0f}")

                st.markdown("#### Actual Cost (AC)")
                st.latex(r"AC = \text{Actual Cost of Work Performed}")
                st.info(f"**Your Value:** AC = {metrics['AC']:,.0f}")

            with col2:
                st.markdown("#### Budget at Completion (BAC)")
                st.latex(r"BAC = \text{Total Planned Budget}")
                st.info(f"**Your Value:** BAC = {metrics['BAC']:,.0f}")

                st.markdown("#### Percent Complete (Planned)")
                st.latex(r"\%Complete_{planned} = \frac{PV}{BAC} \times 100")
                st.info(f"**Calculation:** ({metrics['PV']:,.0f} / {metrics['BAC']:,.0f}) × 100 = **{metrics['PC_planned']:.1f}%**")

                st.markdown("#### Percent Complete (Earned)")
                st.latex(r"\%Complete_{earned} = \frac{EV}{BAC} \times 100")
                st.info(f"**Calculation:** ({metrics['EV']:,.0f} / {metrics['BAC']:,.0f}) × 100 = **{metrics['PC_earned']:.1f}%**")

            st.markdown("---")
            st.markdown("### Variance Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Schedule Variance (SV)")
                st.latex(r"SV = EV - PV")
                st.info(f"**Calculation:** {metrics['EV']:,.0f} - {metrics['PV']:,.0f} = **{metrics['SV']:,.0f}**")
                if metrics['SV'] >= 0:
                    st.success("SV ≥ 0: Project is ahead of or on schedule")
                else:
                    st.error("SV < 0: Project is behind schedule")

            with col2:
                st.markdown("#### Cost Variance (CV)")
                st.latex(r"CV = EV - AC")
                st.info(f"**Calculation:** {metrics['EV']:,.0f} - {metrics['AC']:,.0f} = **{metrics['CV']:,.0f}**")
                if metrics['CV'] >= 0:
                    st.success("CV ≥ 0: Project is under or on budget")
                else:
                    st.error("CV < 0: Project is over budget")

            st.markdown("---")
            st.markdown("### Performance Indices")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Schedule Performance Index (SPI)")
                st.latex(r"SPI = \frac{EV}{PV}")
                st.info(f"**Calculation:** {metrics['EV']:,.0f} / {metrics['PV']:,.0f} = **{metrics['SPI']:.3f}**")
                if metrics['SPI'] >= 1:
                    st.success("SPI ≥ 1: Getting more work done than planned")
                else:
                    st.error(f"SPI < 1: Only {metrics['SPI']*100:.1f}% of planned work rate")

            with col2:
                st.markdown("#### Cost Performance Index (CPI)")
                st.latex(r"CPI = \frac{EV}{AC}")
                st.info(f"**Calculation:** {metrics['EV']:,.0f} / {metrics['AC']:,.0f} = **{metrics['CPI']:.3f}**")
                if metrics['CPI'] >= 1:
                    st.success("CPI ≥ 1: Getting more value per unit spent")
                else:
                    st.error(f"CPI < 1: Only {metrics['CPI']:.2f} value per unit spent")

            st.markdown("---")
            st.markdown("### Forecasting")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Estimate at Completion (EAC) - Typical Variance")
                st.latex(r"EAC = \frac{BAC}{CPI}")
                st.info(f"**Calculation:** {metrics['BAC']:,.0f} / {metrics['CPI']:.3f} = **{metrics['EAC_typical']:,.0f}**")
                st.caption("Assumes current cost performance will continue")

                st.markdown("#### Estimate at Completion (EAC) - Atypical Variance")
                st.latex(r"EAC = AC + (BAC - EV)")
                st.info(f"**Calculation:** {metrics['AC']:,.0f} + ({metrics['BAC']:,.0f} - {metrics['EV']:,.0f}) = **{metrics['EAC_atypical']:,.0f}**")
                st.caption("Assumes remaining work at original budget rate")

                st.markdown("#### Estimate at Completion (EAC) - Combined")
                st.latex(r"EAC = AC + \frac{BAC - EV}{CPI \times SPI}")
                st.info(f"**Calculation:** {metrics['AC']:,.0f} + ({metrics['BAC']:,.0f} - {metrics['EV']:,.0f}) / ({metrics['CPI']:.3f} × {metrics['SPI']:.3f}) = **{metrics['EAC_combined']:,.0f}**")
                st.caption("Considers both cost and schedule performance")

            with col2:
                st.markdown("#### Estimate to Complete (ETC)")
                st.latex(r"ETC = EAC - AC")
                st.info(f"**Calculation:** {metrics['EAC_typical']:,.0f} - {metrics['AC']:,.0f} = **{metrics['ETC_typical']:,.0f}**")
                st.caption("Amount needed to complete the project")

                st.markdown("#### Variance at Completion (VAC)")
                st.latex(r"VAC = BAC - EAC")
                st.info(f"**Calculation:** {metrics['BAC']:,.0f} - {metrics['EAC_typical']:,.0f} = **{metrics['VAC']:,.0f}**")
                if metrics['VAC'] >= 0:
                    st.success("VAC ≥ 0: Expected to finish under budget")
                else:
                    st.error("VAC < 0: Expected to exceed budget")

                st.markdown("#### To-Complete Performance Index (TCPI)")
                st.latex(r"TCPI_{BAC} = \frac{BAC - EV}{BAC - AC}")
                st.info(f"**Calculation:** ({metrics['BAC']:,.0f} - {metrics['EV']:,.0f}) / ({metrics['BAC']:,.0f} - {metrics['AC']:,.0f}) = **{metrics['TCPI_BAC']:.3f}**")
                if metrics['TCPI_BAC'] <= 1.1:
                    st.success("TCPI ≤ 1.1: Target is achievable")
                else:
                    st.warning("TCPI > 1.1: Target may be difficult to achieve")

        # ==================== TAB 3: CHARTS ====================
        with tab3:
            st.header("Visual Analysis")

            if periods_data is not None and len(periods_data) > 0:
                # Classic EVM Chart with Cost & Schedule Variance
                st.subheader("Cost & Schedule Variance (Classic EVM Chart)")

                # Find current period (last period with data)
                current_period_idx = len(periods_data)
                for i in range(len(periods_data) - 1, -1, -1):
                    if periods_data['EV_cumulative'].iloc[i] > 0:
                        current_period_idx = i + 1
                        break

                fig_classic = create_classic_evm_chart(
                    planned_value_data=periods_data['PV_cumulative'].tolist(),
                    actual_cost_data=periods_data['AC_cumulative'].tolist(),
                    earned_value_data=periods_data['EV_cumulative'].tolist(),
                    current_period=current_period_idx,
                    bac=bac,
                    eac=metrics['EAC_typical']
                )
                st.plotly_chart(fig_classic, use_container_width=True)

                st.markdown("""
                **Understanding the EVM Chart (PMBOK/IPMA Standard):**

                **Main Curves:**
                - **PV (BCWS)** - Planned Value / Budgeted Cost of Work Scheduled (blue)
                - **EV (BCWP)** - Earned Value / Budgeted Cost of Work Performed (green)
                - **AC (ACWP)** - Actual Cost / Actual Cost of Work Performed (red)
                - **EAC** - Estimate at Completion forecast (cyan dashed)

                **Key Indicators:**
                - **Cost Variance (CV)** = EV - AC: Vertical gap between EV and AC
                - **Schedule Variance (SV)** = EV - PV: Vertical gap between EV and PV
                - **SV-time** - Schedule variance expressed in time units
                - **BAC** - Budget at Completion (original budget)
                - **EAC** - Estimate at Completion (forecasted final cost)
                - **Management Reserve** - Buffer between BAC and EAC (if over budget)

                **Interpretation:**
                - CV > 0: Under budget | CV < 0: Over budget
                - SV > 0: Ahead of schedule | SV < 0: Behind schedule
                """)

                st.markdown("---")

                # Variance Chart
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Variance Trend")
                    fig_variance = create_variance_chart(periods_data)
                    st.plotly_chart(fig_variance, use_container_width=True)

                with col2:
                    st.subheader("Performance Indices Trend")
                    fig_indices = create_index_chart(periods_data)
                    st.plotly_chart(fig_indices, use_container_width=True)

            # Forecast Comparison (always show)
            st.subheader("Budget vs. Forecast Comparison")
            fig_forecast = create_forecast_chart(metrics['BAC'], metrics['EAC_typical'],
                                                 metrics['AC'], metrics['EV'])
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Quadrant Analysis
            st.subheader("Project Status Quadrant")

            fig_quadrant = go.Figure()

            # Add quadrant backgrounds
            fig_quadrant.add_shape(type="rect", x0=0, y0=1, x1=1, y1=2,
                                  fillcolor="rgba(255,200,200,0.3)", line_width=0)
            fig_quadrant.add_shape(type="rect", x0=1, y0=1, x1=2, y1=2,
                                  fillcolor="rgba(200,255,200,0.3)", line_width=0)
            fig_quadrant.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                                  fillcolor="rgba(255,200,200,0.5)", line_width=0)
            fig_quadrant.add_shape(type="rect", x0=1, y0=0, x1=2, y1=1,
                                  fillcolor="rgba(255,255,200,0.3)", line_width=0)

            # Add current position
            fig_quadrant.add_trace(go.Scatter(
                x=[metrics['CPI']], y=[metrics['SPI']],
                mode='markers+text',
                marker=dict(size=20, color='blue', symbol='star'),
                text=['Current Status'],
                textposition='top center',
                name='Project Status'
            ))

            # Add reference lines
            fig_quadrant.add_hline(y=1, line_dash="dash", line_color="gray")
            fig_quadrant.add_vline(x=1, line_dash="dash", line_color="gray")

            # Add quadrant labels
            fig_quadrant.add_annotation(x=0.5, y=1.5, text="Behind Schedule<br>Under Budget",
                                       showarrow=False, font=dict(size=12))
            fig_quadrant.add_annotation(x=1.5, y=1.5, text="Ahead of Schedule<br>Under Budget",
                                       showarrow=False, font=dict(size=12, color="green"))
            fig_quadrant.add_annotation(x=0.5, y=0.5, text="Behind Schedule<br>Over Budget",
                                       showarrow=False, font=dict(size=12, color="red"))
            fig_quadrant.add_annotation(x=1.5, y=0.5, text="Ahead of Schedule<br>Over Budget",
                                       showarrow=False, font=dict(size=12))

            fig_quadrant.update_layout(
                title="Project Status Quadrant (SPI vs CPI)",
                xaxis_title="Cost Performance Index (CPI)",
                yaxis_title="Schedule Performance Index (SPI)",
                xaxis=dict(range=[0, 2]),
                yaxis=dict(range=[0, 2]),
                height=500,
                showlegend=False
            )

            st.plotly_chart(fig_quadrant, use_container_width=True)

        # ==================== TAB 4: INTERPRETATIONS ====================
        with tab4:
            st.header("Results Interpretation")

            st.markdown("### Overall Project Health Assessment")

            # Overall status
            if metrics['SPI'] >= 1 and metrics['CPI'] >= 1:
                st.success("🎉 **EXCELLENT**: Project is ahead of schedule AND under budget!")
            elif metrics['SPI'] >= 1 and metrics['CPI'] < 1:
                st.warning("⚡ **ATTENTION NEEDED**: Project is ahead of schedule but over budget. Cost control measures recommended.")
            elif metrics['SPI'] < 1 and metrics['CPI'] >= 1:
                st.warning("⏰ **ATTENTION NEEDED**: Project is under budget but behind schedule. Schedule recovery actions recommended.")
            else:
                st.error("🚨 **CRITICAL**: Project is behind schedule AND over budget. Immediate corrective actions required!")

            st.markdown("---")
            st.markdown("### Detailed Metric Interpretations")

            # Schedule Variance
            interpretation, status = interpret_metric('SV', metrics['SV'])
            st.markdown(f"#### Schedule Variance (SV) = {metrics['SV']:,.0f}")
            if status == "good":
                st.success(interpretation)
            elif status == "bad":
                st.error(interpretation)
            else:
                st.warning(interpretation)

            # Cost Variance
            interpretation, status = interpret_metric('CV', metrics['CV'])
            st.markdown(f"#### Cost Variance (CV) = {metrics['CV']:,.0f}")
            if status == "good":
                st.success(interpretation)
            elif status == "bad":
                st.error(interpretation)
            else:
                st.warning(interpretation)

            # SPI
            interpretation, status = interpret_metric('SPI', metrics['SPI'])
            st.markdown(f"#### Schedule Performance Index (SPI) = {metrics['SPI']:.3f}")
            if status == "good":
                st.success(interpretation)
            elif status == "bad":
                st.error(interpretation)
            else:
                st.warning(interpretation)

            # CPI
            interpretation, status = interpret_metric('CPI', metrics['CPI'])
            st.markdown(f"#### Cost Performance Index (CPI) = {metrics['CPI']:.3f}")
            if status == "good":
                st.success(interpretation)
            elif status == "bad":
                st.error(interpretation)
            else:
                st.warning(interpretation)

            # Forecast Analysis
            st.markdown("---")
            st.markdown("### Forecast Analysis")

            st.markdown(f"#### Estimate at Completion (EAC) = {metrics['EAC_typical']:,.0f}")
            variance_pct = ((metrics['EAC_typical'] - metrics['BAC']) / metrics['BAC']) * 100
            if metrics['EAC_typical'] <= metrics['BAC']:
                st.success(f"✅ **Good News**: The project is forecasted to complete {metrics['BAC'] - metrics['EAC_typical']:,.0f} ({abs(variance_pct):.1f}%) under the original budget.")
            else:
                st.error(f"⚠️ **Warning**: The project is forecasted to exceed the original budget by {metrics['EAC_typical'] - metrics['BAC']:,.0f} ({variance_pct:.1f}%). Consider scope reduction or additional funding.")

            # TCPI Analysis
            st.markdown(f"#### To-Complete Performance Index (TCPI) = {metrics['TCPI_BAC']:.3f}")
            interpretation, status = interpret_metric('TCPI', metrics['TCPI_BAC'])
            if status == "good":
                st.success(interpretation)
            else:
                st.error(interpretation)

            # Recommendations
            st.markdown("---")
            st.markdown("### 📋 Recommendations")

            recommendations = []

            if metrics['SPI'] < 1:
                recommendations.append("**Schedule Recovery**: Consider adding resources, working overtime, or fast-tracking activities to recover schedule.")
            if metrics['CPI'] < 1:
                recommendations.append("**Cost Control**: Review and reduce non-essential expenses, negotiate with vendors, or consider value engineering.")
            if metrics['TCPI_BAC'] > 1.2:
                recommendations.append("**Re-baseline**: The required future performance may be unrealistic. Consider re-baselining the project with stakeholder approval.")
            if metrics['SPI'] >= 1 and metrics['CPI'] >= 1:
                recommendations.append("**Maintain Performance**: Continue current management practices. Monitor for any emerging issues.")
            if metrics['VAC'] < 0:
                recommendations.append("**Request Additional Funding**: If cost overrun cannot be avoided, prepare a change request for additional budget.")

            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

        # ==================== TAB 5: THEORY ====================
        with tab5:
            st.header("📚 EVM Theory & IPMA Guidelines")

            st.markdown("""
            ### What is Earned Value Management (EVM)?

            Earned Value Management is a project management technique that integrates **scope**, **schedule**,
            and **cost** measurements to assess project performance and progress. It provides early warning
            signals of performance problems while there is still time for corrective action.

            ### IPMA Competence Framework

            According to the **International Project Management Association (IPMA)**, EVM is a key competence
            under the **Practice Competence Elements**, specifically:

            ### Key EVM Concepts

            #### The Three Pillars of EVM
            """)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                **📅 Planned Value (PV)**

                Also known as BCWS (Budgeted Cost of Work Scheduled)

                - The authorized budget for scheduled work
                - Represents the baseline plan
                - Used to measure schedule variance
                """)

            with col2:
                st.markdown("""
                **✅ Earned Value (EV)**

                Also known as BCWP (Budgeted Cost of Work Performed)

                - The value of work actually completed
                - Measured against the baseline
                - The core of EVM analysis
                """)

            with col3:
                st.markdown("""
                **💰 Actual Cost (AC)**

                Also known as ACWP (Actual Cost of Work Performed)

                - The actual cost incurred
                - Compared against EV for cost analysis
                - Must align with EV measurements
                """)

            st.markdown("""
            ---
            ### Understanding Variances

            | Metric | Formula | Positive Value | Negative Value |
            |--------|---------|----------------|----------------|
            | Schedule Variance (SV) | EV - PV | Ahead of schedule | Behind schedule |
            | Cost Variance (CV) | EV - AC | Under budget | Over budget |
            | Variance at Completion (VAC) | BAC - EAC | Under budget | Over budget |

            ---
            ### Understanding Performance Indices

            | Index | Formula | Value > 1 | Value < 1 |
            |-------|---------|-----------|-----------|
            | Schedule Performance Index (SPI) | EV ÷ PV | Ahead of schedule | Behind schedule |
            | Cost Performance Index (CPI) | EV ÷ AC | Under budget | Over budget |
            | To-Complete Performance Index (TCPI) | (BAC-EV) ÷ (BAC-AC) | Easier to achieve | Harder to achieve |

            ---
            ### Forecasting Methods

            There are three common methods for calculating **Estimate at Completion (EAC)**:

            1. **EAC = BAC / CPI** (Typical Variance)
               - Used when current cost variance is expected to continue
               - Most commonly used method

            2. **EAC = AC + (BAC - EV)** (Atypical Variance)
               - Used when current variance is considered atypical
               - Assumes remaining work at original budget rate

            3. **EAC = AC + (BAC - EV) / (CPI × SPI)** (Combined Index)
               - Considers both cost and schedule performance
               - More conservative estimate

            ---
            ### Best Practices for EVM Implementation

            1. **Define the Work Breakdown Structure (WBS)** clearly
            2. **Establish the baseline** before project execution
            3. **Measure progress objectively** using earned value methods:
               - 0/100 (Complete/Incomplete)
               - 50/50 (Start/Complete)
               - Percent Complete
               - Milestone Weights
            4. **Report regularly** (weekly or bi-weekly)
            5. **Analyze variances** and take corrective action
            6. **Update forecasts** as new data becomes available

            ---
            ### When to Use Each EAC Formula

            | Scenario | Recommended Formula |
            |----------|---------------------|
            | Variance expected to continue | EAC = BAC / CPI |
            | One-time issue (atypical) | EAC = AC + (BAC - EV) |
            | Schedule impacts costs | EAC = AC + (BAC - EV) / (CPI × SPI) |
            | Management estimate available | EAC = AC + Bottom-up ETC |

            ---
            ### References

            - IPMA Individual Competence Baseline (ICB4)
            - PMI Practice Standard for Earned Value Management
            - ISO 21508:2018 - Earned Value Management in Project and Programme Management
            """)

    else:
        # Welcome message when no calculation has been done
        st.info("👈 Enter your project data in the sidebar and click **Calculate EVM Metrics** to begin.")

        st.markdown("""
        ### Welcome to the EVM Calculator!

        #### What you can do:

        1. **Single Period Analysis**: Enter current PV, EV, and AC values
        2. **Multi-Period Analysis**: Track project performance over multiple periods
        3. **Use Example Data**: Load sample data to explore the tool

        #### What you'll learn:

        - How to calculate EVM metrics (SV, CV, SPI, CPI)
        - How to interpret performance indices
        - How to forecast project completion (EAC, ETC, VAC)
        - How to visualize project performance with S-curves
        """)

        # Quick reference
        st.markdown("### Quick Reference: EVM Formulas")

        col1, col2 = st.columns(2)

        with col1:
            st.latex(r"SV = EV - PV")
            st.latex(r"CV = EV - AC")
            st.latex(r"SPI = \frac{EV}{PV}")
            st.latex(r"CPI = \frac{EV}{AC}")

        with col2:
            st.latex(r"EAC = \frac{BAC}{CPI}")
            st.latex(r"ETC = EAC - AC")
            st.latex(r"VAC = BAC - EAC")
            st.latex(r"TCPI = \frac{BAC - EV}{BAC - AC}")

    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 20px; color: #666;">
            <p style="margin-bottom: 10px;">
                <strong>Developed by Boris Taliev</strong>
            </p>
            <p style="margin-bottom: 10px;">
                <a href="https://www.linkedin.com/in/boris-taliev-a9960a6b/" target="_blank" style="text-decoration: none; color: #0077B5;">
                    🔗 LinkedIn Profile
                </a>
            </p>
            <p style="margin-bottom: 5px;">
                <strong>My other projects:</strong>
            </p>
            <p style="margin-bottom: 5px;">
                <a href="https://rccalcs-tp7b2fnba2jsmk8jasxtxv.streamlit.app" target="_blank" style="text-decoration: none; color: #FF4B4B;">
                    📊 RC Calculator
                </a>
                &nbsp;|&nbsp;
                <a href="https://steelsheet.streamlit.app" target="_blank" style="text-decoration: none; color: #FF4B4B;">
                    🔩 Steel Sheet Calculator
                </a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
