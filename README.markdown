# Financial Report Generator

## Overview
The **Financial Report Generator** is a Streamlit-based web application designed to automate the creation of comprehensive financial reports. Inspired by tools like FinAlyzer and Power BI, it consolidates financial data from CSV files, performs analysis, generates visualizations, and produces a downloadable PDF report. The application supports year-over-year comparisons, KPI alerts, variance analysis, and predictive revenue forecasts using linear regression.

Key features include:
- **Data Input**: Upload CSV files for current and previous year financial data.
- **Analysis**: Calculates total revenue, expenses, profit, and profit margins, with optional entity-level and budget variance analysis.
- **Visualizations**: Generates professional charts (line, bar, pie, area, box, and table) using Matplotlib and Seaborn.
- **PDF Report**: Produces a FinAlyzer-inspired PDF report with tables, narrative text, and embedded visualizations.
- **KPI Alerts**: Flags months with profit margins below a user-defined threshold.
- **Predictive Analytics**: Forecasts next year's revenue using linear regression.

## Prerequisites
To run this project, you need the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- A compatible web browser (e.g., Chrome, Firefox)

## Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd financial-report-generator
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required Python packages using the provided `requirements.txt` or manually:
   ```bash
   pip install streamlit pandas matplotlib seaborn scikit-learn reportlab numpy
   ```

   Alternatively, save the following to `requirements.txt`:
   ```
   streamlit==1.38.0
   pandas==2.2.2
   matplotlib==3.9.2
   seaborn==0.13.2
   scikit-learn==1.5.1
   reportlab==4.2.2
   numpy==1.26.4
   ```

4. **Save the Code**:
   Save the provided Python script as `app.py` in your project directory.

## Usage
1. **Run the Application**:
   Start the Streamlit app by running:
   ```bash
   streamlit run app.py
   ```
   This will launch a local web server, and a browser window will open (typically at `http://localhost:8501`).

2. **Upload Data**:
   - Prepare CSV files for current and previous year financial data with at least the following columns: `Month`, `Revenue`, `Expenses`, `Profit`.
   - Optional columns: `Entity` (for entity-level analysis) and `Budget` (for variance analysis).
   - Upload the CSV files using the file uploader in the app.

3. **Configure Settings**:
   - Enter the company name (default: "Example Inc.").
   - Set the profit margin threshold for KPI alerts using the slider (default: 30%).

4. **Generate Report**:
   - Once both CSV files are uploaded, the app will:
     - Display a text-based report with financial summaries, KPI alerts, and predictions.
     - Show visualizations (line chart, bar chart, pie chart, area chart, box plot, and table).
     - Provide a button to download the report as a PDF.

5. **Sample CSV Format**:
   ```csv
   Month,Revenue,Expenses,Profit,Entity,Budget
   January,100000,60000,40000,Division A,95000
   February,120000,70000,50000,Division A,110000
   ...
   ```

## Features
- **Streamlit Interface**: User-friendly web interface for data upload and configuration.
- **Financial Analysis**:
  - Consolidates data across entities (if provided).
  - Performs year-over-year comparisons and variance analysis (if budget data is available).
  - Flags low profit margins with KPI alerts.
- **Visualizations**:
  - Revenue trend (line chart)
  - Profit comparison (bar chart)
  - Expense breakdown (pie chart)
  - Revenue vs. expenses (area chart)
  - Profit distribution (box plot)
  - Monthly financial details (table)
- **PDF Report**: Professional report with tables, narrative text, visualizations, and an audit trail.
- **Predictive Modeling**: Uses linear regression to forecast next year's revenue based on historical data.

## Limitations
- Requires CSV files with specific columns (`Month`, `Revenue`, `Expenses`, `Profit`).
- Predictive modeling assumes linear trends, which may not account for complex market dynamics.
- Visualizations are saved as temporary PNG files and deleted after PDF generation.
- Entity and budget analysis are optional and depend on the presence of `Entity` and `Budget` columns in the CSV.

## Troubleshooting
- **CSV Errors**: Ensure CSV files have the required columns and valid numerical data.
- **Dependency Issues**: Verify all packages are installed correctly. Use `pip list` to check versions.
- **PDF Generation**: Ensure sufficient disk space for temporary image files and check for `reportlab` compatibility.
- **Streamlit Not Loading**: Run `streamlit run app.py` from the correct directory and ensure no port conflicts.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, please contact pprathames1010@gmail.com or open an issue on the repository.

---
