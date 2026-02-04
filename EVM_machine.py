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

# Page configuration
st.set_page_config(
    page_title="EVM Calculator - IPMA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .formula-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    .interpretation-good {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    .interpretation-bad {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    .interpretation-neutral {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
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
            'negative': ("⚠️ **Inefficient Cost Performance**: For every dollar spent, only ${:.2f} of value is being earned.".format(value), "bad"),
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
        yaxis_title='Cost ($)',
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
    fig.update_yaxes(title_text="Variance ($)", row=1, col=1)
    fig.update_yaxes(title_text="Variance ($)", row=2, col=1)

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

    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors, text=[f'${v:,.0f}' for v in values],
               textposition='outside')
    ])

    fig.update_layout(
        title='Budget vs. Forecast Comparison',
        yaxis_title='Cost ($)',
        height=400,
        showlegend=False
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

            bac = st.number_input("Budget at Completion (BAC) $",
                                  min_value=0.0, value=100000.0, step=1000.0,
                                  help="Total planned budget for the entire project")

            pv = st.number_input("Planned Value (PV) $",
                                 min_value=0.0, value=50000.0, step=1000.0,
                                 help="Budgeted cost of work scheduled to date")

            ev = st.number_input("Earned Value (EV) $",
                                 min_value=0.0, value=45000.0, step=1000.0,
                                 help="Budgeted cost of work actually completed")

            ac = st.number_input("Actual Cost (AC) $",
                                 min_value=0.0, value=55000.0, step=1000.0,
                                 help="Actual cost incurred for work completed")

            periods_data = None

        else:  # Multi-Period Analysis
            st.subheader("Project Setup")

            bac = st.number_input("Budget at Completion (BAC) $",
                                  min_value=0.0, value=100000.0, step=1000.0)

            periods = st.number_input("Number of Periods",
                                      min_value=2, max_value=24, value=6, step=1)

            st.subheader("Enter Period Data")

            # Create input dataframe
            periods_data = pd.DataFrame({
                'Period': range(1, periods + 1),
                'PV': [0.0] * periods,
                'EV': [0.0] * periods,
                'AC': [0.0] * periods
            })

            edited_df = st.data_editor(
                periods_data,
                column_config={
                    "Period": st.column_config.NumberColumn("Period", disabled=True),
                    "PV": st.column_config.NumberColumn("Planned Value ($)", min_value=0, format="$%.0f"),
                    "EV": st.column_config.NumberColumn("Earned Value ($)", min_value=0, format="$%.0f"),
                    "AC": st.column_config.NumberColumn("Actual Cost ($)", min_value=0, format="$%.0f"),
                },
                hide_index=True,
                use_container_width=True
            )

            periods_data = edited_df.copy()

            # Calculate cumulative values
            periods_data['PV_cumulative'] = periods_data['PV'].cumsum()
            periods_data['EV_cumulative'] = periods_data['EV'].cumsum()
            periods_data['AC_cumulative'] = periods_data['AC'].cumsum()

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
                st.metric("Budget at Completion (BAC)", f"${metrics['BAC']:,.0f}")
            with col2:
                st.metric("Planned Value (PV)", f"${metrics['PV']:,.0f}",
                         f"{metrics['PC_planned']:.1f}% complete")
            with col3:
                st.metric("Earned Value (EV)", f"${metrics['EV']:,.0f}",
                         f"{metrics['PC_earned']:.1f}% complete")
            with col4:
                st.metric("Actual Cost (AC)", f"${metrics['AC']:,.0f}")

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
                st.metric("Schedule Variance (SV)", f"${metrics['SV']:,.0f}", sv_delta)

            with col2:
                cv_delta = "Under Budget" if metrics['CV'] >= 0 else "Over Budget"
                st.metric("Cost Variance (CV)", f"${metrics['CV']:,.0f}", cv_delta)

            with col3:
                vac_delta = "Savings" if metrics['VAC'] >= 0 else "Overrun"
                st.metric("Variance at Completion (VAC)", f"${metrics['VAC']:,.0f}", vac_delta)

            # Forecasts
            st.subheader("Forecasts")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Estimate at Completion (EAC)", f"${metrics['EAC_typical']:,.0f}",
                         f"${metrics['EAC_typical'] - metrics['BAC']:+,.0f} vs BAC")
            with col2:
                st.metric("Estimate to Complete (ETC)", f"${metrics['ETC_typical']:,.0f}")
            with col3:
                st.metric("To-Complete Performance Index", f"{metrics['TCPI_BAC']:.2f}",
                         "Required CPI for remaining work")

        # ==================== TAB 2: FORMULAS ====================
        with tab2:
            st.header("Formulas & Calculations")

            st.markdown("### Basic EVM Metrics")

            # PV, EV, AC
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Planned Value (PV)")
                st.latex(r"PV = \text{Budgeted Cost of Work Scheduled}")
                st.info(f"**Your Value:** PV = ${metrics['PV']:,.0f}")

                st.markdown("#### Earned Value (EV)")
                st.latex(r"EV = \text{Budgeted Cost of Work Performed}")
                st.info(f"**Your Value:** EV = ${metrics['EV']:,.0f}")

                st.markdown("#### Actual Cost (AC)")
                st.latex(r"AC = \text{Actual Cost of Work Performed}")
                st.info(f"**Your Value:** AC = ${metrics['AC']:,.0f}")

            with col2:
                st.markdown("#### Budget at Completion (BAC)")
                st.latex(r"BAC = \text{Total Planned Budget}")
                st.info(f"**Your Value:** BAC = ${metrics['BAC']:,.0f}")

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
                st.info(f"**Calculation:** ${metrics['EV']:,.0f} - ${metrics['PV']:,.0f} = **${metrics['SV']:,.0f}**")
                if metrics['SV'] >= 0:
                    st.success("SV ≥ 0: Project is ahead of or on schedule")
                else:
                    st.error("SV < 0: Project is behind schedule")

            with col2:
                st.markdown("#### Cost Variance (CV)")
                st.latex(r"CV = EV - AC")
                st.info(f"**Calculation:** ${metrics['EV']:,.0f} - ${metrics['AC']:,.0f} = **${metrics['CV']:,.0f}**")
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
                st.info(f"**Calculation:** ${metrics['EV']:,.0f} / ${metrics['PV']:,.0f} = **{metrics['SPI']:.3f}**")
                if metrics['SPI'] >= 1:
                    st.success("SPI ≥ 1: Getting more work done than planned")
                else:
                    st.error(f"SPI < 1: Only {metrics['SPI']*100:.1f}% of planned work rate")

            with col2:
                st.markdown("#### Cost Performance Index (CPI)")
                st.latex(r"CPI = \frac{EV}{AC}")
                st.info(f"**Calculation:** ${metrics['EV']:,.0f} / ${metrics['AC']:,.0f} = **{metrics['CPI']:.3f}**")
                if metrics['CPI'] >= 1:
                    st.success("CPI ≥ 1: Getting more value per dollar spent")
                else:
                    st.error(f"CPI < 1: Only ${metrics['CPI']:.2f} value per dollar spent")

            st.markdown("---")
            st.markdown("### Forecasting")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Estimate at Completion (EAC) - Typical Variance")
                st.latex(r"EAC = \frac{BAC}{CPI}")
                st.info(f"**Calculation:** ${metrics['BAC']:,.0f} / {metrics['CPI']:.3f} = **${metrics['EAC_typical']:,.0f}**")
                st.caption("Assumes current cost performance will continue")

                st.markdown("#### Estimate at Completion (EAC) - Atypical Variance")
                st.latex(r"EAC = AC + (BAC - EV)")
                st.info(f"**Calculation:** ${metrics['AC']:,.0f} + (${metrics['BAC']:,.0f} - ${metrics['EV']:,.0f}) = **${metrics['EAC_atypical']:,.0f}**")
                st.caption("Assumes remaining work at original budget rate")

                st.markdown("#### Estimate at Completion (EAC) - Combined")
                st.latex(r"EAC = AC + \frac{BAC - EV}{CPI \times SPI}")
                st.info(f"**Calculation:** ${metrics['AC']:,.0f} + (${metrics['BAC']:,.0f} - ${metrics['EV']:,.0f}) / ({metrics['CPI']:.3f} × {metrics['SPI']:.3f}) = **${metrics['EAC_combined']:,.0f}**")
                st.caption("Considers both cost and schedule performance")

            with col2:
                st.markdown("#### Estimate to Complete (ETC)")
                st.latex(r"ETC = EAC - AC")
                st.info(f"**Calculation:** ${metrics['EAC_typical']:,.0f} - ${metrics['AC']:,.0f} = **${metrics['ETC_typical']:,.0f}**")
                st.caption("Amount needed to complete the project")

                st.markdown("#### Variance at Completion (VAC)")
                st.latex(r"VAC = BAC - EAC")
                st.info(f"**Calculation:** ${metrics['BAC']:,.0f} - ${metrics['EAC_typical']:,.0f} = **${metrics['VAC']:,.0f}**")
                if metrics['VAC'] >= 0:
                    st.success("VAC ≥ 0: Expected to finish under budget")
                else:
                    st.error("VAC < 0: Expected to exceed budget")

                st.markdown("#### To-Complete Performance Index (TCPI)")
                st.latex(r"TCPI_{BAC} = \frac{BAC - EV}{BAC - AC}")
                st.info(f"**Calculation:** (${metrics['BAC']:,.0f} - ${metrics['EV']:,.0f}) / (${metrics['BAC']:,.0f} - ${metrics['AC']:,.0f}) = **{metrics['TCPI_BAC']:.3f}**")
                if metrics['TCPI_BAC'] <= 1.1:
                    st.success("TCPI ≤ 1.1: Target is achievable")
                else:
                    st.warning("TCPI > 1.1: Target may be difficult to achieve")

        # ==================== TAB 3: CHARTS ====================
        with tab3:
            st.header("Visual Analysis")

            if periods_data is not None and len(periods_data) > 0:
                # S-Curve
                st.subheader("S-Curve Analysis")
                fig_s_curve = create_s_curve(periods_data)
                st.plotly_chart(fig_s_curve, use_container_width=True)

                st.markdown("""
                **How to read the S-Curve:**
                - **PV (Blue)**: The planned progress baseline
                - **EV (Green)**: Actual progress in terms of value earned
                - **AC (Red)**: Actual costs incurred
                - If EV is below PV, the project is behind schedule
                - If AC is above EV, the project is over budget
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
            st.markdown(f"#### Schedule Variance (SV) = ${metrics['SV']:,.0f}")
            if status == "good":
                st.markdown(f'<div class="interpretation-good">{interpretation}</div>', unsafe_allow_html=True)
            elif status == "bad":
                st.markdown(f'<div class="interpretation-bad">{interpretation}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="interpretation-neutral">{interpretation}</div>', unsafe_allow_html=True)

            # Cost Variance
            interpretation, status = interpret_metric('CV', metrics['CV'])
            st.markdown(f"#### Cost Variance (CV) = ${metrics['CV']:,.0f}")
            if status == "good":
                st.markdown(f'<div class="interpretation-good">{interpretation}</div>', unsafe_allow_html=True)
            elif status == "bad":
                st.markdown(f'<div class="interpretation-bad">{interpretation}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="interpretation-neutral">{interpretation}</div>', unsafe_allow_html=True)

            # SPI
            interpretation, status = interpret_metric('SPI', metrics['SPI'])
            st.markdown(f"#### Schedule Performance Index (SPI) = {metrics['SPI']:.3f}")
            if status == "good":
                st.markdown(f'<div class="interpretation-good">{interpretation}</div>', unsafe_allow_html=True)
            elif status == "bad":
                st.markdown(f'<div class="interpretation-bad">{interpretation}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="interpretation-neutral">{interpretation}</div>', unsafe_allow_html=True)

            # CPI
            interpretation, status = interpret_metric('CPI', metrics['CPI'])
            st.markdown(f"#### Cost Performance Index (CPI) = {metrics['CPI']:.3f}")
            if status == "good":
                st.markdown(f'<div class="interpretation-good">{interpretation}</div>', unsafe_allow_html=True)
            elif status == "bad":
                st.markdown(f'<div class="interpretation-bad">{interpretation}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="interpretation-neutral">{interpretation}</div>', unsafe_allow_html=True)

            # Forecast Analysis
            st.markdown("---")
            st.markdown("### Forecast Analysis")

            st.markdown(f"#### Estimate at Completion (EAC) = ${metrics['EAC_typical']:,.0f}")
            variance_pct = ((metrics['EAC_typical'] - metrics['BAC']) / metrics['BAC']) * 100
            if metrics['EAC_typical'] <= metrics['BAC']:
                st.markdown(f'<div class="interpretation-good">✅ **Good News**: The project is forecasted to complete ${metrics["BAC"] - metrics["EAC_typical"]:,.0f} ({abs(variance_pct):.1f}%) under the original budget.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="interpretation-bad">⚠️ **Warning**: The project is forecasted to exceed the original budget by ${metrics["EAC_typical"] - metrics["BAC"]:,.0f} ({variance_pct:.1f}%). Consider scope reduction or additional funding.</div>', unsafe_allow_html=True)

            # TCPI Analysis
            st.markdown(f"#### To-Complete Performance Index (TCPI) = {metrics['TCPI_BAC']:.3f}")
            interpretation, status = interpret_metric('TCPI', metrics['TCPI_BAC'])
            if status == "good":
                st.markdown(f'<div class="interpretation-good">{interpretation}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="interpretation-bad">{interpretation}</div>', unsafe_allow_html=True)

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

            - **Practice 4.5.6**: Plan and control resources and costs
            - **Practice 4.5.7**: Plan and control schedule

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

        This educational tool helps you understand and apply **Earned Value Management (EVM)**
        according to IPMA standards.

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


if __name__ == "__main__":
    main()
