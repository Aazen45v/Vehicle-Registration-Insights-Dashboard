# Vehicle Registration Insights Dashboard

This interactive dashboard provides an investor-focused view of vehicle registration trends in India, built with Python and Streamlit. It allows for dynamic filtering by date range and vehicle category to analyze market share and performance trends.

**Video Walkthrough:** [**INSERT YOUR VIDEO WALKTHROUGH LINK HERE**]

**Live Application Link:** [**INSERT YOUR DEPLOYED APP LINK HERE, E.G., FROM STREAMLIT COMMUNITY CLOUD**]

---

## 🚀 Features

- **Interactive Filtering**: Dynamically filter data by a specific date range, vehicle category, and manufacturer.
- **KPI Metrics**: At-a-glance metrics for total registrations, period-over-period growth, and year-on-year growth.
- **Dynamic Visualizations**:
    - A time-series line chart showing daily registration trends for the selected period.
    - An interactive donut chart displaying market share for the top 10 manufacturers.
- **Detailed Performance Table**: A sortable table breaking down manufacturer performance, including current registrations and YoY growth comparisons.
- **Professional UI/UX**: A clean, modern interface with custom styling for a polished user experience.

---

## 🛠️ Setup and Installation

To run this application locally, please follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [INSERT YOUR GITHUB REPO LINK HERE]
    cd [YOUR REPO NAME]
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Ensure you have the `requirements.txt` file in your project directory and run:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```
    streamlit
    pandas
    numpy
    plotly
    ```

4.  **Place the Data File**
    Download the dataset (`vahan_data_2025.csv`) and place it in the root directory of the project, alongside the `app.py` script.

5.  **Run the Application**
    In your terminal, navigate to the project's root directory and run the following command:
    ```bash
    streamlit run app.py
    ```
    The application should now be open and accessible in your web browser.

---

## 📊 Data Assumptions

The dashboard is designed to work with a data snapshot that represents cumulative registrations up to a certain point. Due to the nature of this snapshot, several assumptions were made to enable time-based analysis:

1.  **Current Data Snapshot**: The provided `vahan_data_2025.csv` file is treated as the dataset for the most recent year (which we have designated as 2025 for simulation purposes).

2.  **Historical Data Simulation**: To enable critical features like Year-on-Year (YoY) and Period-over-Period (PoP) growth analysis, historical data for the preceding two years (2023 and 2024) is programmatically simulated. This is done by applying a random reduction factor (between 8% and 15%) to the previous year's numbers.

3.  **Daily Data Simulation**: To power the date range selector, the annual registration data for each manufacturer and vehicle category is distributed across the 365 days of its respective year. This distribution is randomized to simulate daily fluctuations in registration activity.

These simulations are essential for demonstrating the full analytical capabilities of the dashboard, which would otherwise be impossible with a single-point-in-time data file.

---

## 🗺️ Feature Roadmap

If this project were to be continued, the following features would be on the roadmap for future development:

-   **Real-time Data Integration**: Connect the dashboard to a live database or API to fetch real-time registration data, removing the need for data simulation.
-   **Geospatial Analysis**: Add a map visualization to show registration data by state or region, allowing for geographical trend analysis.
-   **Advanced Forecasting**: Integrate time-series forecasting models (like ARIMA or Prophet) to predict future registration trends based on historical data.
-   **User Authentication & Saved Views**: Allow users to create accounts, save their preferred filter settings, and create custom reports.
-   **Export Functionality**: Add buttons to export charts as images (PNG/JPEG) and data tables as CSV or Excel files for offline analysis and reporting.
-   **Fuel Type Analysis**: Incorporate data on fuel types (Petrol, Diesel, EV, etc.) to analyze the market shift towards electric vehicles and other alternative fuels.

