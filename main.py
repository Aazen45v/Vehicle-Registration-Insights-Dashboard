# ==============================================================================
# VEHICLE REGISTRATION INSIGHTS DASHBOARD (V3.6 - Uniform Multiselect)
#
# Author: [Your Name]
# Assignment: Backend Developer Internship
#
# Description:
# This version adds custom CSS to make the selected items in the multiselect
# filter have a uniform width for a cleaner sidebar interface.
#
# Usage Instructions:
# 1. Save this script as `app.py`.
# 2. Ensure `vahan_data_2025.csv` is located in the same directory.
# 3. Install dependencies via `pip install -r requirements.txt`.
# 4. Launch the app with the command: `streamlit run app.py`
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vehicle Registration Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- THEME & STYLING ---
PRIMARY_COLOR = "#1f77b4"  # Muted Blue
SECONDARY_COLOR = "#ff7f0e"  # Safety Orange
TEXT_COLOR = "#31333F"
BACKGROUND_COLOR = "#FFFFFF"
CHART_TEMPLATE = "plotly_white"  # Clean chart theme


# --- DATA LOADING AND PROCESSING ---
@st.cache_data
def load_and_process_data(file_path: str) -> pd.DataFrame:
    """
    Load, clean, and transform vehicle registration data.
    Simulate historical and quarterly data for comprehensive trend analysis.

    Args:
        file_path (str): Path to the source CSV file.

    Returns:
        pd.DataFrame: Processed and expanded vehicle registration dataset.
    """
    # Verify data file exists
    if not Path(file_path).is_file():
        st.error(f"Data file not found: `{file_path}`. Please place `vahan_data_2025.csv` in the app directory.")
        return pd.DataFrame()

    try:
        # Load CSV, skipping multi-header rows, rename columns
        df = pd.read_csv(file_path, skiprows=4).iloc[:, 1:]
        df.columns = [
            'Manufacturer', '2WIC', '2WN', '2WT', '3WIC', '3WN', '3WT', '4WIC',
            'HGV', 'HMV', 'HPV', 'LGV', 'LMV', 'LPV', 'MGV', 'MMV', 'MPV', 'OTH', 'TOTAL'
        ]
        df.dropna(subset=['Manufacturer'], inplace=True)

        # Convert numeric columns from string to int, cleaning commas
        for col in df.columns:
            if col != 'Manufacturer':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

        # Reshape wide to long format for analysis
        df_long = df.melt(
            id_vars=['Manufacturer'],
            value_vars=[col for col in df.columns if col not in ['Manufacturer', 'TOTAL']],
            var_name='Vehicle_Code',
            value_name='Registrations'
        )
        df_long['Year'] = 2025

        # Simulate previous years data (2023, 2024) for trend comparison
        df_2024 = df_long.copy()
        df_2024['Year'] = 2024
        df_2024['Registrations'] = (df_2024['Registrations'] * np.random.uniform(0.85, 0.92, size=len(df_2024))).astype(
            int)

        df_2023 = df_2024.copy()
        df_2023['Year'] = 2023
        df_2023['Registrations'] = (df_2023['Registrations'] * np.random.uniform(0.85, 0.92, size=len(df_2023))).astype(
            int)

        # Combine all years
        df_full = pd.concat([df_2023, df_2024, df_long], ignore_index=True)

        # Map vehicle codes to categories and filter relevant categories
        category_map = {
            '2WIC': 'Two-Wheeler', '2WN': 'Two-Wheeler', '2WT': 'Two-Wheeler',
            '3WIC': 'Three-Wheeler', '3WN': 'Three-Wheeler', '3WT': 'Three-Wheeler',
            '4WIC': 'Four-Wheeler', 'LMV': 'Four-Wheeler', 'MPV': 'Four-Wheeler',
        }
        df_full['Vehicle Category'] = df_full['Vehicle_Code'].map(category_map).fillna('Other')
        df_full = df_full[df_full['Vehicle Category'] != 'Other']

        # Distribute yearly registrations into quarterly figures with seasonality
        quarters_data = []
        for _, row in df_full.iterrows():
            total = row['Registrations']
            q1 = int(total * 0.28)
            q2 = int(total * 0.22)
            q3 = int(total * 0.20)
            q4 = total - (q1 + q2 + q3)
            quarters_data.append({'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4})

        quarters_df = pd.DataFrame(quarters_data)
        df_full.reset_index(drop=True, inplace=True)
        df_full = pd.concat([df_full, quarters_df], axis=1)

        return df_full

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return pd.DataFrame()


# --- GROWTH CALCULATION ---
def calculate_growth(current: float, previous: float) -> float:
    """Calculate percentage growth between current and previous values."""
    if previous == 0 or previous is None:
        return 0.0
    return (current - previous) / previous


# --- SIDEBAR FILTERS ---
def display_sidebar(df: pd.DataFrame) -> dict:
    """Render sidebar controls and return user selections."""
    st.sidebar.title("Dashboard Controls")
    st.sidebar.markdown("---")

    years = sorted(df['Year'].unique(), reverse=True)
    selected_year = st.sidebar.selectbox("Select Year", years, help="Select the analysis year.")

    selected_quarter = st.sidebar.selectbox("Select Quarter", ["All", "Q1", "Q2", "Q3", "Q4"],
                                            help="Select the quarter to analyze; 'All' aggregates the full year.")

    temp_df = df[df['Year'] == selected_year]
    categories = sorted(temp_df['Vehicle Category'].unique())
    selected_categories = st.sidebar.multiselect("Filter by Vehicle Category", categories, default=categories)

    filtered_df = temp_df[temp_df['Vehicle Category'].isin(selected_categories)] if selected_categories else temp_df
    manufacturers = sorted(filtered_df['Manufacturer'].unique())
    selected_manufacturers = st.sidebar.multiselect("Filter by Manufacturer(s)", manufacturers, default=[])

    st.sidebar.markdown("---")

    return {
        "year": selected_year,
        "quarter": selected_quarter,
        "categories": selected_categories,
        "manufacturers": selected_manufacturers,
    }


# --- KPI DISPLAY ---
def display_kpi_metrics(filtered_df: pd.DataFrame, base_df: pd.DataFrame, selections: dict) -> None:
    """Calculate and present key performance indicators."""
    st.subheader(f"Key Metrics for {selections['quarter']} {selections['year']}")

    current_total = filtered_df['Registrations'].sum()

    # Quarter-on-quarter growth (except for Q1 and 'All')
    qoq_growth = 0.0
    if selections['quarter'] not in ['All', 'Q1']:
        prev_q = f"q{int(selections['quarter'][1]) - 1}"
        prev_q_df = base_df[base_df['Year'] == selections['year']]
        if selections['categories']:
            prev_q_df = prev_q_df[prev_q_df['Vehicle Category'].isin(selections['categories'])]
        if selections['manufacturers']:
            prev_q_df = prev_q_df[prev_q_df['Manufacturer'].isin(selections['manufacturers'])]
        prev_q_total = prev_q_df[prev_q].sum() if not prev_q_df.empty else 0
        qoq_growth = calculate_growth(current_total, prev_q_total)

    # Year-on-year growth
    prev_y_df = base_df[base_df['Year'] == selections['year'] - 1]
    if selections['categories']:
        prev_y_df = prev_y_df[prev_y_df['Vehicle Category'].isin(selections['categories'])]
    if selections['manufacturers']:
        prev_y_df = prev_y_df[prev_y_df['Manufacturer'].isin(selections['manufacturers'])]

    if selections['quarter'] != 'All':
        prev_y_total = prev_y_df[selections['quarter'].lower()].sum() if not prev_y_df.empty else 0
    else:
        prev_y_total = prev_y_df['Registrations'].sum()

    yoy_growth = calculate_growth(current_total, prev_y_total)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Total Registrations ({selections['quarter']}, {selections['year']})", f"{current_total:,}")
    with col2:
        st.metric(
            "Quarter-on-Quarter Growth",
            f"{qoq_growth:.2%}",
            delta=f"{qoq_growth:.2%}" if qoq_growth else None,
            help="Growth compared to the previous quarter of the same year."
        )
    with col3:
        st.metric(
            "Year-on-Year Growth",
            f"{yoy_growth:.2%}",
            delta=f"{yoy_growth:.2%}" if yoy_growth else None,
            help="Growth compared to the same quarter (or full year) of the previous year."
        )


# --- VISUALIZATIONS ---
def display_charts(filtered_df: pd.DataFrame, base_df: pd.DataFrame, selections: dict) -> None:
    """Render main data visualizations for registration trends and market share."""
    col1, col2 = st.columns((6, 4))

    with col1:
        st.subheader("Yearly Registration Trends")

        yearly_df = base_df.copy()
        if selections['categories']:
            yearly_df = yearly_df[yearly_df['Vehicle Category'].isin(selections['categories'])]
        if selections['manufacturers']:
            yearly_df = yearly_df[yearly_df['Manufacturer'].isin(selections['manufacturers'])]

        yearly_summary = yearly_df.groupby('Year')['Registrations'].sum().reset_index()

        fig = px.bar(
            yearly_summary,
            x='Year', y='Registrations',
            text='Registrations',
            title="Total Registrations by Year",
            color_discrete_sequence=[PRIMARY_COLOR] * len(yearly_summary),
            template=CHART_TEMPLATE,
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_layout(
            title_font_size=18,
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title=None,
            yaxis_title="Total Registrations",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Manufacturer Market Share")

        market_share = filtered_df.groupby('Manufacturer')['Registrations'].sum().nlargest(10).reset_index()
        total_regs = market_share['Registrations'].sum()

        fig_pie = px.pie(
            market_share,
            names='Manufacturer',
            values='Registrations',
            title=f"Top 10 Manufacturers ({selections['quarter']}, {selections['year']})",
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.Blues_r,
            template=CHART_TEMPLATE,
        )
        fig_pie.update_layout(
            title_font_size=18,
            title_x=0.5,
            annotations=[dict(text=f'{total_regs:,}<br>Total', x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05] * len(market_share))
        st.plotly_chart(fig_pie, use_container_width=True)


# --- PERFORMANCE TABLE ---
def display_performance_table(base_df: pd.DataFrame, selections: dict) -> None:
    """Render a detailed manufacturer performance table with YoY growth percentage."""
    st.subheader("Manufacturer Performance Breakdown")

    df_perf = base_df.copy()
    if selections['categories']:
        df_perf = df_perf[df_perf['Vehicle Category'].isin(selections['categories'])]

    if selections['quarter'] != 'All':
        df_perf['Current_Period_Reg'] = df_perf[selections['quarter'].lower()]
    else:
        df_perf['Current_Period_Reg'] = df_perf['Registrations']

    current_perf = df_perf[df_perf['Year'] == selections['year']].groupby('Manufacturer')['Current_Period_Reg'].sum()
    prev_perf = df_perf[df_perf['Year'] == selections['year'] - 1].groupby('Manufacturer')['Current_Period_Reg'].sum()

    perf_df = pd.DataFrame({
        f'Registrations ({selections["year"]})': current_perf,
        f'Registrations ({selections["year"] - 1})': prev_perf,
    }).fillna(0).astype(int)

    perf_df['YoY Growth'] = perf_df.apply(lambda row: calculate_growth(row.iloc[0], row.iloc[1]), axis=1)

    # Clip values for the progress bar visualization, ensuring they are between 0 and 1 (0% to 100%)
    # This prevents errors with negative growth or growth > 100% in the progress bar.
    # The original value is still used for the text display via the format string.
    perf_df['YoY Growth Bar'] = perf_df['YoY Growth'].clip(0, 1)

    perf_df.sort_values(by=f'Registrations ({selections["year"]})', ascending=False, inplace=True)

    st.dataframe(
        perf_df,
        use_container_width=True,
        column_config={
            f'Registrations ({selections["year"]})': st.column_config.NumberColumn(format="%,.0f"),
            f'Registrations ({selections["year"] - 1})': st.column_config.NumberColumn(format="%,.0f"),
            "YoY Growth": st.column_config.NumberColumn(
                "YoY Growth",
                format="%.2f%%"
            ),
            "YoY Growth Bar": st.column_config.ProgressColumn(
                "YoY Growth",
                format="%.2f%%",
                min_value=0,
                max_value=1,
            ),
        },
        # Hide the helper column from view
        column_order=[col for col in perf_df.columns if col != 'YoY Growth Bar']
    )


# --- MAIN APP ---
def main():
    # Use st.markdown to inject HTML for a larger, centered title and custom CSS
    st.markdown("""
        <style>
            .title-container {
                text-align: center;
                margin-bottom: 2rem;
            }
            .title-container h1 {
                font-size: 3.5rem;
                font-weight: 700;
            }
            .title-container p {
                font-size: 1.25rem;
                color: #555;
            }
            /* Hide the sidebar's scrollbar */
            [data-testid="stSidebar"] > div:first-child {
                overflow-y: hidden;
            }
            /* Style for multiselect items */
            .stMultiSelect [data-baseweb="tag"] {
                width: 100%;
                display: flex;
                justify-content: space-between;
            }
        </style>
        <div class="title-container">
            <h1>Vehicle Registration Insights</h1>
            <p>An investor-focused view of vehicle registration trends in India.</p>
        </div>
    """, unsafe_allow_html=True)

    data = load_and_process_data('vahan_data_2025.csv')
    if data.empty:
        st.warning("Failed to load data. Please verify the file and try again.")
        st.stop()

    selections = display_sidebar(data)

    filtered = data[data['Year'] == selections['year']].copy()
    if selections['quarter'] != 'All':
        filtered['Registrations'] = filtered[selections['quarter'].lower()]
    else:
        filtered = filtered.groupby(['Manufacturer', 'Year', 'Vehicle Category'])['Registrations'].sum().reset_index()

    if selections['categories']:
        filtered = filtered[filtered['Vehicle Category'].isin(selections['categories'])]
    if selections['manufacturers']:
        filtered = filtered[filtered['Manufacturer'].isin(selections['manufacturers'])]

    if filtered.empty:
        st.warning("No data found for the selected filters. Please adjust your selections.")
    else:
        display_kpi_metrics(filtered, data, selections)
        st.markdown("---")
        display_charts(filtered, data, selections)
        st.markdown("---")
        display_performance_table(data, selections)


if __name__ == "__main__":
    main()
