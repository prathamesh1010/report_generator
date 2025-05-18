
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# import numpy as np
# from io import StringIO

# # Streamlit app configuration
# st.set_page_config(page_title="Financial Report Generator", layout="wide")

# # Function to load and validate CSV data
# def load_data(file, year):
#     try:
#         df = pd.read_csv(file)
#         required_columns = ['Month', 'Revenue', 'Expenses', 'Profit']
#         if not all(col in df.columns for col in required_columns):
#             st.error(f"CSV for {year} must contain columns: {', '.join(required_columns)}")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error loading {year} data: {e}")
#         return None

# # Function to generate financial report
# def generate_report(current_df, previous_df, company_name):
#     report = StringIO()
#     report.write(f"# Financial Report for {company_name}\n\n")
    
#     # Calculate key metrics
#     current_totals = {
#         'Revenue': current_df['Revenue'].sum(),
#         'Expenses': current_df['Expenses'].sum(),
#         'Profit': current_df['Profit'].sum()
#     }
#     previous_totals = {
#         'Revenue': previous_df['Revenue'].sum(),
#         'Expenses': previous_df['Expenses'].sum(),
#         'Profit': previous_df['Profit'].sum()
#     }
    
#     # Write summary to report
#     report.write("## Summary\n")
#     report.write("**Current Year:**\n")
#     for key, value in current_totals.items():
#         report.write(f"- Total {key}: ${value:,.2f}\n")
#     report.write("\n**Previous Year:**\n")
#     for key, value in previous_totals.items():
#         report.write(f"- Total {key}: ${value:,.2f}\n")
    
#     # Year-over-Year Comparison
#     report.write("\n## Year-over-Year Comparison\n")
#     for key in current_totals:
#         change = ((current_totals[key] - previous_totals[key]) / previous_totals[key]) * 100
#         report.write(f"- {key} Change: {change:+.2f}%\n")
    
#     # Predictions for next year
#     X = np.array([1, 2]).reshape(-1, 1)
#     y = np.array([previous_totals['Revenue'], current_totals['Revenue']])
#     model = LinearRegression()
#     model.fit(X, y)
#     next_year_revenue = model.predict(np.array([[3]]))[0]
#     report.write("\n## Predictions\n")
#     report.write(f"- Predicted Revenue for Next Year: ${next_year_revenue:,.2f}\n")
    
#     return report.getvalue()

# # Function to generate visualizations
# def generate_visualizations(current_df, previous_df, company_name):
#     plt.figure(figsize=(10, 6))
#     plt.plot(current_df['Month'], current_df['Revenue'], label='Current Year Revenue', marker='o')
#     plt.plot(previous_df['Month'], previous_df['Revenue'], label='Previous Year Revenue', marker='s')
#     plt.title(f'Revenue Trend - {company_name}')
#     plt.xlabel('Month')
#     plt.ylabel('Revenue ($)')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     st.pyplot(plt)
#     plt.close()
    
#     plt.figure(figsize=(10, 6))
#     bar_width = 0.35
#     months = current_df['Month']
#     x = np.arange(len(months))
#     plt.bar(x - bar_width/2, current_df['Profit'], bar_width, label='Current Year Profit')
#     plt.bar(x + bar_width/2, previous_df['Profit'], bar_width, label='Previous Year Profit')
#     plt.title(f'Profit Comparison - {company_name}')
#     plt.xlabel('Month')
#     plt.ylabel('Profit ($)')
#     plt.xticks(x, months, rotation=45)
#     plt.legend()
#     st.pyplot(plt)
#     plt.close()
    
#     plt.figure(figsize=(8, 8))
#     expenses = current_df['Expenses'].sum()
#     other = current_df['Revenue'].sum() - expenses - current_df['Profit'].sum()
#     plt.pie([expenses, other], labels=['Expenses', 'Other'], autopct='%1.1f%%', startangle=140)
#     plt.title(f'Expense Breakdown (Current Year) - {company_name}')
#     st.pyplot(plt)
#     plt.close()

# # Streamlit UI
# st.title("Financial Report Generator")
# st.write("Upload this year's and previous year's financial data (CSV) to generate a report.")

# company_name = st.text_input("Enter Company Name", "Example Inc.")
# current_year_file = st.file_uploader("Upload Current Year Financial Data (CSV)", type="csv")
# previous_year_file = st.file_uploader("Upload Previous Year Financial Data (CSV)", type="csv")

# if current_year_file and previous_year_file:
#     current_df = load_data(current_year_file, "Current Year")
#     previous_df = load_data(previous_year_file, "Previous Year")
    
#     if current_df is not None and previous_df is not None:
#         st.subheader("Generated Report")
#         report_content = generate_report(current_df, previous_df, company_name)
#         st.text_area("Report Content", report_content, height=400)
        
#         st.download_button(
#             label="Download Report",
#             data=report_content,
#             file_name=f"{company_name}_financial_report.txt",
#             mime="text/plain"
#         )
        
#         st.subheader("Visualizations")
#         generate_visualizations(current_df, previous_df, company_name)
# else:
#     st.info("Please upload both CSV files to generate the report.")
















# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# import numpy as np
# from io import StringIO, BytesIO
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
# from reportlab.lib import colors
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib.units import inch
# import os

# # Streamlit app configuration
# st.set_page_config(page_title="Financial Report Generator", layout="wide")

# # Function to load and validate CSV data
# def load_data(file, year):
#     try:
#         df = pd.read_csv(file)
#         required_columns = ['Month', 'Revenue', 'Expenses', 'Profit']
#         if not all(col in df.columns for col in required_columns):
#             st.error(f"CSV for {year} must contain columns: {', '.join(required_columns)}")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error loading {year} data: {e}")
#         return None

# # Function to generate financial report text
# def generate_report_text(current_df, previous_df, company_name):
#     report = StringIO()
#     report.write(f"# Financial Report for {company_name}\n\n")
    
#     # Calculate key metrics
#     current_totals = {
#         'Revenue': current_df['Revenue'].sum(),
#         'Expenses': current_df['Expenses'].sum(),
#         'Profit': current_df['Profit'].sum()
#     }
#     previous_totals = {
#         'Revenue': previous_df['Revenue'].sum(),
#         'Expenses': previous_df['Expenses'].sum(),
#         'Profit': previous_df['Profit'].sum()
#     }
    
#     # Write summary to report
#     report.write("## Summary\n")
#     report.write("**Current Year:**\n")
#     for key, value in current_totals.items():
#         report.write(f"- Total {key}: ${value:,.2f}\n")
#     report.write("\n**Previous Year:**\n")
#     for key, value in previous_totals.items():
#         report.write(f"- Total {key}: ${value:,.2f}\n")
    
#     # Year-over-Year Comparison
#     report.write("\n## Year-over-Year Comparison\n")
#     for key in current_totals:
#         change = ((current_totals[key] - previous_totals[key]) / previous_totals[key]) * 100
#         report.write(f"- {key} Change: {change:+.2f}%\n")
    
#     # Predictions for next year
#     X = np.array([1, 2]).reshape(-1, 1)
#     y = np.array([previous_totals['Revenue'], current_totals['Revenue']])
#     model = LinearRegression()
#     model.fit(X, y)
#     next_year_revenue = model.predict(np.array([[3]]))[0]
#     revenue_change = ((next_year_revenue - current_totals['Revenue']) / current_totals['Revenue']) * 100
#     report.write("\n## Predictions\n")
#     report.write(f"- Predicted Revenue for Next Year: ${next_year_revenue:,.2f}\n")
#     report.write(f"- Predicted Revenue Change: {revenue_change:+.2f}%\n")
    
#     return report.getvalue(), current_totals, previous_totals, next_year_revenue, revenue_change

# # Function to generate visualizations and save as PNG
# def generate_visualizations(current_df, previous_df, company_name):
#     sns.set_style("whitegrid")
#     image_paths = []
    
#     # Line Chart: Revenue Trend
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(x='Month', y='Revenue', data=current_df, label='Current Year Revenue', marker='o', color='blue', linewidth=2)
#     sns.lineplot(x='Month', y='Revenue', data=previous_df, label='Previous Year Revenue', marker='s', color='orange', linewidth=2)
#     plt.title(f'Revenue Trend - {company_name}', fontsize=14, weight='bold')
#     plt.xlabel('Month', fontsize=12)
#     plt.ylabel('Revenue ($)', fontsize=12)
#     plt.legend(fontsize=10)
#     plt.xticks(rotation=45, fontsize=10)
#     plt.tight_layout()
#     revenue_trend_path = 'revenue_trend.png'
#     plt.savefig(revenue_trend_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(revenue_trend_path)
    
#     # Bar Chart: Profit Comparison
#     plt.figure(figsize=(10, 6))
#     bar_width = 0.35
#     months = current_df['Month']
#     x = np.arange(len(months))
#     plt.bar(x - bar_width/2, current_df['Profit'], bar_width, label='Current Year Profit', color='green')
#     plt.bar(x + bar_width/2, previous_df['Profit'], bar_width, label='Previous Year Profit', color='red')
#     plt.title(f'Profit Comparison - {company_name}', fontsize=14, weight='bold')
#     plt.xlabel('Month', fontsize=12)
#     plt.ylabel('Profit ($)', fontsize=12)
#     plt.xticks(x, months, rotation=45, fontsize=10)
#     plt.legend(fontsize=10)
#     plt.tight_layout()
#     profit_comparison_path = 'profit_comparison.png'
#     plt.savefig(profit_comparison_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(profit_comparison_path)
    
#     # Pie Chart: Expense Breakdown (Current Year)
#     plt.figure(figsize=(8, 8))
#     expenses = current_df['Expenses'].sum()
#     other = current_df['Revenue'].sum() - expenses - current_df['Profit'].sum()
#     plt.pie([expenses, other], labels=['Expenses', 'Other'], autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'])
#     plt.title(f'Expense Breakdown (Current Year) - {company_name}', fontsize=14, weight='bold')
#     plt.tight_layout()
#     expense_breakdown_path = 'expense_breakdown.png'
#     plt.savefig(expense_breakdown_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(expense_breakdown_path)
    
#     return image_paths

# # Function to generate PDF report
# def generate_pdf_report(company_name, report_text, image_paths, output_filename='financial_report.pdf'):
#     buffer = BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=letter)
#     elements = []
#     styles = getSampleStyleSheet()
    
#     # Title
#     elements.append(Paragraph(f"Financial Report for {company_name}", styles['Title']))
#     elements.append(Spacer(1, 12))
    
#     # Report Content
#     for line in report_text.split('\n'):
#         if line.startswith('# '):
#             elements.append(Paragraph(line[2:], styles['Heading1']))
#         elif line.startswith('## '):
#             elements.append(Paragraph(line[3:], styles['Heading2']))
#         elif line.startswith('- '):
#             elements.append(Paragraph(line[2:], styles['Normal']))
#         else:
#             elements.append(Paragraph(line, styles['Normal']))
#         elements.append(Spacer(1, 6))
    
#     # Graphs
#     elements.append(Paragraph("Visualizations", styles['Heading2']))
#     elements.append(Spacer(1, 12))
#     for image_path in image_paths:
#         img = Image(image_path, width=6*inch, height=4*inch)
#         elements.append(img)
#         elements.append(Spacer(1, 12))
    
#     # Build PDF
#     doc.build(elements)
    
#     # Clean up image files
#     for image_path in image_paths:
#         if os.path.exists(image_path):
#             os.remove(image_path)
    
#     buffer.seek(0)
#     return buffer

# # Streamlit UI
# st.title("Financial Report Generator")
# st.write("Upload this year's and previous year's financial data (CSV) to generate a report.")

# company_name = st.text_input("Enter Company Name", "Example Inc.")
# current_year_file = st.file_uploader("Upload Current Year Financial Data (CSV)", type="csv")
# previous_year_file = st.file_uploader("Upload Previous Year Financial Data (CSV)", type="csv")

# if current_year_file and previous_year_file:
#     current_df = load_data(current_year_file, "Current Year")
#     previous_df = load_data(previous_year_file, "Previous Year")
    
#     if current_df is not None and previous_df is not None:
#         # Generate report text and metrics
#         report_content, current_totals, previous_totals, next_year_revenue, revenue_change = generate_report_text(current_df, previous_df, company_name)
        
#         # Display report
#         st.subheader("Generated Report")
#         st.text_area("Report Content", report_content, height=400)
        
#         # Generate visualizations
#         st.subheader("Visualizations")
#         image_paths = generate_visualizations(current_df, previous_df, company_name)
#         for image_path in image_paths:
#             st.image(image_path)
        
#         # Generate and download PDF
#         pdf_buffer = generate_pdf_report(company_name, report_content, image_paths)
#         st.download_button(
#             label="Download Report as PDF",
#             data=pdf_buffer,
#             file_name=f"{company_name}_financial_report.pdf",
#             mime="application/pdf"
#         )
# else:
#     st.info("Please upload both CSV files to generate the report.")












# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# import numpy as np
# from io import StringIO, BytesIO
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
# from reportlab.lib import colors
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch
# from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
# from reportlab.pdfgen import canvas
# import os
# from datetime import datetime

# # Streamlit app configuration
# st.set_page_config(page_title="Financial Report Generator", layout="wide")

# # Function to load and validate CSV data
# def load_data(file, year):
#     try:
#         df = pd.read_csv(file)
#         required_columns = ['Month', 'Revenue', 'Expenses', 'Profit']
#         if not all(col in df.columns for col in required_columns):
#             st.error(f"CSV for {year} must contain columns: {', '.join(required_columns)}")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error loading {year} data: {e}")
#         return None

# # Function to generate financial report text and metrics
# def generate_report_text(current_df, previous_df, company_name):
#     report = StringIO()
    
#     # Executive Summary
#     report.write(f"Executive Summary\n")
#     report.write(f"This report provides a comprehensive analysis of {company_name}'s financial performance for the current and previous years. It includes key metrics, year-over-year comparisons, predictive insights, and visualizations to support strategic decision-making.\n\n")
    
#     # Summary
#     report.write(f"Summary\n")
#     report.write(f"The following section outlines the total financial metrics for {company_name} across the current and previous years.\n")
    
#     # Calculate key metrics
#     current_totals = {
#         'Revenue': current_df['Revenue'].sum(),
#         'Expenses': current_df['Expenses'].sum(),
#         'Profit': current_df['Profit'].sum()
#     }
#     previous_totals = {
#         'Revenue': previous_df['Revenue'].sum(),
#         'Expenses': previous_df['Expenses'].sum(),
#         'Profit': previous_df['Profit'].sum()
#     }
    
#     report.write("Current Year:\n")
#     for key, value in current_totals.items():
#         report.write(f"- Total {key}: ${value:,.2f}\n")
#     report.write("\nPrevious Year:\n")
#     for key, value in previous_totals.items():
#         report.write(f"- Total {key}: ${value:,.2f}\n")
    
#     # Year-over-Year Comparison
#     report.write("\nYear-over-Year Comparison\n")
#     report.write(f"This section compares the financial performance of {company_name} between the current and previous years to identify trends and growth patterns.\n")
#     for key in current_totals:
#         change = ((current_totals[key] - previous_totals[key]) / previous_totals[key]) * 100
#         report.write(f"- {key} Change: {change:+.2f}%\n")
    
#     # Predictions
#     report.write("\nPredictions\n")
#     report.write(f"Using a linear regression model, the following predictions estimate {company_name}'s revenue for the next year based on historical trends.\n")
#     X = np.array([1, 2]).reshape(-1, 1)
#     y = np.array([previous_totals['Revenue'], current_totals['Revenue']])
#     model = LinearRegression()
#     model.fit(X, y)
#     next_year_revenue = model.predict(np.array([[3]]))[0]
#     revenue_change = ((next_year_revenue - current_totals['Revenue']) / current_totals['Revenue']) * 100
#     report.write(f"- Predicted Revenue for Next Year: ${next_year_revenue:,.2f}\n")
#     report.write(f"- Predicted Revenue Change: {revenue_change:+.2f}%\n")
    
#     return report.getvalue(), current_totals, previous_totals, next_year_revenue, revenue_change

# # Function to generate visualizations and save as PNG
# def generate_visualizations(current_df, previous_df, company_name):
#     sns.set_style("whitegrid")
#     image_paths = []
    
#     # Line Chart: Revenue Trend
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(x='Month', y='Revenue', data=current_df, label='Current Year Revenue', marker='o', color='#1f77b4', linewidth=2)
#     sns.lineplot(x='Month', y='Revenue', data=previous_df, label='Previous Year Revenue', marker='s', color='#ff7f0e', linewidth=2)
#     plt.title(f'Revenue Trend - {company_name}', fontsize=14, weight='bold', pad=10)
#     plt.xlabel('Month', fontsize=12)
#     plt.ylabel('Revenue ($)', fontsize=12)
#     plt.legend(fontsize=10)
#     plt.xticks(rotation=45, fontsize=10)
#     plt.tight_layout()
#     revenue_trend_path = 'revenue_trend.png'
#     plt.savefig(revenue_trend_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(revenue_trend_path)
    
#     # Bar Chart: Profit Comparison
#     plt.figure(figsize=(10, 6))
#     bar_width = 0.35
#     months = current_df['Month']
#     x = np.arange(len(months))
#     plt.bar(x - bar_width/2, current_df['Profit'], bar_width, label='Current Year Profit', color='#2ca02c')
#     plt.bar(x + bar_width/2, previous_df['Profit'], bar_width, label='Previous Year Profit', color='#d62728')
#     plt.title(f'Profit Comparison - {company_name}', fontsize=14, weight='bold', pad=10)
#     plt.xlabel('Month', fontsize=12)
#     plt.ylabel('Profit ($)', fontsize=12)
#     plt.xticks(x, months, rotation=45, fontsize=10)
#     plt.legend(fontsize=10)
#     plt.tight_layout()
#     profit_comparison_path = 'profit_comparison.png'
#     plt.savefig(profit_comparison_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(profit_comparison_path)
    
#     # Pie Chart: Expense Breakdown (Current Year)
#     plt.figure(figsize=(8, 8))
#     expenses = current_df['Expenses'].sum()
#     other = current_df['Revenue'].sum() - expenses - current_df['Profit'].sum()
#     plt.pie([expenses, other], labels=['Expenses', 'Other'], autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'], textprops={'fontsize': 12})
#     plt.title(f'Expense Breakdown (Current Year) - {company_name}', fontsize=14, weight='bold', pad=10)
#     plt.tight_layout()
#     expense_breakdown_path = 'expense_breakdown.png'
#     plt.savefig(expense_breakdown_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(expense_breakdown_path)
    
#     # Area Chart: Revenue vs Expenses (Current Year)
#     plt.figure(figsize=(10, 6))
#     plt.stackplot(current_df['Month'], current_df['Revenue'], current_df['Expenses'], labels=['Revenue', 'Expenses'], colors=['#1f77b4', '#ff7f0e'])
#     plt.title(f'Revenue vs Expenses (Current Year) - {company_name}', fontsize=14, weight='bold', pad=10)
#     plt.xlabel('Month', fontsize=12)
#     plt.ylabel('Amount ($)', fontsize=12)
#     plt.legend(loc='upper left', fontsize=10)
#     plt.xticks(rotation=45, fontsize=10)
#     plt.tight_layout()
#     area_chart_path = 'area_chart.png'
#     plt.savefig(area_chart_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(area_chart_path)
    
#     # Box Plot: Profit Distribution
#     plt.figure(figsize=(10, 6))
#     profit_data = pd.DataFrame({
#         'Current Year': current_df['Profit'],
#         'Previous Year': previous_df['Profit']
#     })
#     sns.boxplot(data=profit_data, palette=['#2ca02c', '#d62728'])
#     plt.title(f'Profit Distribution - {company_name}', fontsize=14, weight='bold', pad=10)
#     plt.ylabel('Profit ($)', fontsize=12)
#     plt.tight_layout()
#     box_plot_path = 'box_plot.png'
#     plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(box_plot_path)
    
#     # # Heatmap: Correlation Matrix
#     # plt.figure(figsize=(8, 6))
#     # corr_data = current_df[['Revenue', 'Expenses', 'Profit']].corr()
#     # sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f', annot_kws={'fontsize': 12})
#     # plt.title(f'Correlation Matrix (Current Year) - {company_name}', fontsize=14, weight='bold', pad=10)
#     # plt.tight_layout()
#     # heatmap_path = 'heatmap.png'
#     # plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
#     # plt.close()
#     # image_paths.append(heatmap_path)
    
#     return image_paths

# # Function to generate Power BI-inspired and traditional PDF report
# def generate_pdf_report(company_name, report_text, current_totals, previous_totals, next_year_revenue, revenue_change, image_paths, output_filename='financial_report.pdf'):
#     buffer = BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
#     elements = []
#     styles = getSampleStyleSheet()
    
#     # Custom styles
#     header_style = ParagraphStyle(name='Header', fontSize=16, textColor=colors.white, alignment=TA_CENTER, spaceAfter=12)
#     section_style = ParagraphStyle(name='Section', fontSize=12, textColor=colors.black, spaceAfter=8, fontName='Helvetica-Bold')
#     narrative_style = ParagraphStyle(name='Narrative', fontSize=10, textColor=colors.black, spaceAfter=6, alignment=TA_JUSTIFY)
#     normal_style = ParagraphStyle(name='Normal', fontSize=10, textColor=colors.black, spaceAfter=6)
    
#     # Header
#     header_table = Table([[Paragraph(f"Financial Report for {company_name}", header_style)]], colWidths=[7.5*inch])
#     header_table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, -1), colors.darkblue),
#         ('BOX', (0, 0), (-1, -1), 1, colors.black),
#         ('PADDING', (0, 0), (-1, -1), 12)
#     ]))
#     elements.append(header_table)
#     elements.append(Spacer(1, 12))
    
#     # Executive Summary
#     elements.append(Paragraph("Executive Summary", section_style))
#     exec_summary = f"This report provides a comprehensive analysis of {company_name}'s financial performance for the current and previous years. It includes key metrics, year-over-year comparisons, predictive insights, and visualizations to support strategic decision-making."
#     elements.append(Paragraph(exec_summary, narrative_style))
#     elements.append(Spacer(1, 12))
    
#     # Summary Table
#     elements.append(Paragraph("Summary", section_style))
#     elements.append(Paragraph(f"The following table outlines the total financial metrics for {company_name} across the current and previous years.", narrative_style))
#     summary_data = [
#         ['Metric', 'Current Year', 'Previous Year'],
#         ['Revenue', f"${current_totals['Revenue']:,.2f}", f"${previous_totals['Revenue']:,.2f}"],
#         ['Expenses', f"${current_totals['Expenses']:,.2f}", f"${previous_totals['Expenses']:,.2f}"],
#         ['Profit', f"${current_totals['Profit']:,.2f}", f"${previous_totals['Profit']:,.2f}"]
#     ]
#     summary_table = Table(summary_data, colWidths=[2*inch, 2.5*inch, 2.5*inch])
#     summary_table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, 0), 10),
#         ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
#         ('GRID', (0, 0), (-1, -1), 1, colors.black)
#     ]))
#     elements.append(summary_table)
#     elements.append(Spacer(1, 12))
    
#     # Year-over-Year Comparison
#     elements.append(Paragraph("Year-over-Year Comparison", section_style))
#     elements.append(Paragraph(f"This section compares the financial performance of {company_name} between the current and previous years to identify trends and growth patterns.", narrative_style))
#     comparison_data = [['Metric', 'Change (%)']]
#     for key in current_totals:
#         change = ((current_totals[key] - previous_totals[key]) / previous_totals[key]) * 100
#         comparison_data.append([key, f"{change:+.2f}%"])
#     comparison_table = Table(comparison_data, colWidths=[3*inch, 2*inch])
#     comparison_table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, 0), 10),
#         ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
#         ('GRID', (0, 0), (-1, -1), 1, colors.black)
#     ]))
#     elements.append(comparison_table)
#     elements.append(Spacer(1, 12))
    
#     # Predictions
#     elements.append(Paragraph("Predictions", section_style))
#     elements.append(Paragraph(f"Using a linear regression model, the following predictions estimate {company_name}'s revenue for the next year based on historical trends.", narrative_style))
#     prediction_data = [
#         ['Metric', 'Value'],
#         ['Predicted Revenue', f"${next_year_revenue:,.2f}"],
#         ['Predicted Revenue Change', f"{revenue_change:+.2f}%"]
#     ]
#     prediction_table = Table(prediction_data, colWidths=[3*inch, 2*inch])
#     prediction_table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, 0), 10),
#         ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
#         ('GRID', (0, 0), (-1, -1), 1, colors.black)
#     ]))
#     elements.append(prediction_table)
#     elements.append(Spacer(1, 12))
    
#     # Visualizations
#     elements.append(Paragraph("Visualizations", section_style))
#     elements.append(Paragraph(f"The following charts provide a visual representation of {company_name}'s financial performance, including trends, comparisons, and correlations.", narrative_style))
#     elements.append(Spacer(1, 12))
#     for i, image_path in enumerate(image_paths):
#         img = Image(image_path, width=6*inch, height=4*inch)
#         elements.append(img)
#         elements.append(Spacer(1, 12))
#         if (i + 1) % 2 == 0 and i < len(image_paths) - 1:
#             elements.append(PageBreak())
    
#     # Footer with timestamp
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     elements.append(Paragraph(f"Generated on: {timestamp}", normal_style))
    
#     # Build PDF with custom canvas for page numbers
#     def add_page_numbers(canvas, doc):
#         page_num = canvas.getPageNumber()
#         text = f"Page {page_num}"
#         canvas.setFont("Helvetica", 9)
#         canvas.drawRightString(7.5*inch, 0.3*inch, text)
    
#     doc.build(elements, onFirstPage=add_page_numbers, onLaterPages=add_page_numbers)
    
#     # Clean up image files
#     for image_path in image_paths:
#         if os.path.exists(image_path):
#             os.remove(image_path)
    
#     buffer.seek(0)
#     return buffer

# # Streamlit UI
# st.title("Financial Report Generator")
# st.write("Upload this year's and previous year's financial data (CSV) to generate a Power BI-inspired and traditional report.")

# company_name = st.text_input("Enter Company Name", "Example Inc.")
# current_year_file = st.file_uploader("Upload Current Year Financial Data (CSV)", type="csv")
# previous_year_file = st.file_uploader("Upload Previous Year Financial Data (CSV)", type="csv")

# if current_year_file and previous_year_file:
#     current_df = load_data(current_year_file, "Current Year")
#     previous_df = load_data(previous_year_file, "Previous Year")
    
#     if current_df is not None and previous_df is not None:
#         # Generate report text and metrics
#         report_content, current_totals, previous_totals, next_year_revenue, revenue_change = generate_report_text(current_df, previous_df, company_name)
        
#         # Display report
#         st.subheader("Generated Report")
#         st.text_area("Report Content", report_content, height=400)
        
#         # Generate visualizations
#         st.subheader("Visualizations")
#         image_paths = generate_visualizations(current_df, previous_df, company_name)
#         for image_path in image_paths:
#             st.image(image_path)
        
#         # Generate and download PDF
#         pdf_buffer = generate_pdf_report(company_name, report_content, current_totals, previous_totals, next_year_revenue, revenue_change, image_paths)
#         st.download_button(
#             label="Download Report as PDF",
#             data=pdf_buffer,
#             file_name=f"{company_name}_financial_report.pdf",
#             mime="application/pdf"
#         )
# else:
#     st.info("Please upload both CSV files to generate the report.")









import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas
import os
from datetime import datetime


# Streamlit app configuration - must be first command
st.set_page_config(page_title="Financial Report Generator", layout="wide")

# Add a visible header to your app
st.title("Financial Report Generator")

# Function to load and validate CSV data
def load_data(file, year):
    try:
        df = pd.read_csv(file)
        required_columns = ['Month', 'Revenue', 'Expenses', 'Profit']
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

        return df
    except Exception as e:
        st.error(f"Error loading {year} data: {e}")
        return None

# Function to generate financial report text and metrics
def generate_report_text(current_df, previous_df, company_name, kpi_threshold):
    report = StringIO()
    
    # Executive Summary
    report.write(f"Executive Summary\n")
    report.write(f"This report, automated financial report, provides a unified analysis of {company_name}'s financial performance across entities for the current and previous years. It consolidates financial data, delivers actionable insights through KPIs, variance analysis, and predictive forecasts, and supports strategic decision-making with detailed visualizations.\n\n")
    
    # Consolidation and Segment-Wise Analysis
    report.write(f"Financial Consolidation\n")
    report.write(f"Data is consolidated across entities (if provided), automating aggregation of Revenue, Expenses, and Profit. Segment-wise profitability is analyzed by entity and month, with KPI alerts for profit margins below {kpi_threshold*100:.0f}%.\n")
    
    # Calculate key metrics
    current_totals = {
        'Revenue': current_df['Revenue'].sum(),
        'Expenses': current_df['Expenses'].sum(),
        'Profit': current_df['Profit'].sum()
    }
    previous_totals = {
        'Revenue': previous_df['Revenue'].sum(),
        'Expenses': previous_df['Expenses'].sum(),
        'Profit': previous_df['Profit'].sum()
    }
    
    # Entity-Level Analysis
    entity_analysis = ""
    if 'Entity' in current_df.columns:
        entity_summary = current_df.groupby('Entity')[['Revenue', 'Expenses', 'Profit']].sum()
        entity_analysis += "\nEntity-Level Summary (Current Year):\n"
        for entity, row in entity_summary.iterrows():
            profit_margin = (row['Profit'] / row['Revenue'] * 100) if row['Revenue'] > 0 else 0
            entity_analysis += f"- {entity}: Revenue ${row['Revenue']:,.2f}, Expenses ${row['Expenses']:,.2f}, Profit ${row['Profit']:,.2f}, Profit Margin {profit_margin:.2f}%\n"
    
    # KPI Alerts
    current_df['Profit Margin'] = current_df['Profit'] / current_df['Revenue']
    kpi_alerts = current_df[current_df['Profit Margin'] < kpi_threshold][['Month', 'Profit Margin']]
    kpi_alert_text = "\nKPI Alerts (Profit Margin < {:.0f}%):\n".format(kpi_threshold*100)
    if not kpi_alerts.empty:
        for _, row in kpi_alerts.iterrows():
            kpi_alert_text += f"- {row['Month']}: Profit Margin {row['Profit Margin']*100:.2f}%\n"
    else:
        kpi_alert_text += "- No months below the profit margin threshold.\n"
    
    # Variance Analysis
    variance_analysis = ""
    if 'Budget' in current_df.columns:
        current_df['Variance'] = current_df['Revenue'] - current_df['Budget']
        variance_analysis += "\nVariance Analysis (Actual vs. Budget, Current Year):\n"
        for _, row in current_df.iterrows():
            variance_analysis += f"- {row['Month']}: Actual ${row['Revenue']:,.2f}, Budget ${row['Budget']:,.2f}, Variance ${row['Variance']:,.2f}\n"
    
    # Summary
    report.write(f"\nSummary\n")
    report.write(f"This section summarizes {company_name}'s consolidated financial performance, integrating data across all entities.\n")
    report.write("Current Year:\n")
    for key, value in current_totals.items():
        report.write(f"- Total {key}: ${value:,.2f}\n")
    report.write("\nPrevious Year:\n")
    for key, value in previous_totals.items():
        report.write(f"- Total {key}: ${value:,.2f}\n")
    report.write(entity_analysis)
    report.write(kpi_alert_text)
    report.write(variance_analysis)
    
    # Year-over-Year Comparison
    report.write("\nYear-over-Year Comparison\n")
    report.write(f"This analysis compares {company_name}'s performance to identify growth trends and operational efficiencies.\n")
    for key in current_totals:
        change = ((current_totals[key] - previous_totals[key]) / previous_totals[key]) * 100
        report.write(f"- {key} Change: {change:+.2f}%\n")
    
    # Predictions
    report.write("\nPredictions\n")
    report.write(f"Using linear regression, this section forecasts {company_name}'s revenue for the next year, based on historical trends.\n")
    X = np.array([1, 2]).reshape(-1, 1)
    y = np.array([previous_totals['Revenue'], current_totals['Revenue']])
    model = LinearRegression()
    model.fit(X, y)
    next_year_revenue = model.predict(np.array([[3]]))[0]
    revenue_change = ((next_year_revenue - current_totals['Revenue']) / current_totals['Revenue']) * 100
    report.write(f"- Predicted Revenue for Next Year: ${next_year_revenue:,.2f}\n")
    report.write(f"- Predicted Revenue Change: {revenue_change:+.2f}%\n")
    
    # Audit Trail Placeholder
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.write(f"\nAudit Trail\n")
    report.write(f"- Data uploaded and processed on: {timestamp}\n")
    
    return report.getvalue(), current_totals, previous_totals, next_year_revenue, revenue_change, current_df

# Function to generate visualizations and save as PNG
def generate_visualizations(current_df, previous_df, company_name, kpi_threshold):
    sns.set_style("whitegrid")
    image_paths = []
    
    # Line Chart: Revenue Trend
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Month', y='Revenue', data=current_df, label='Current Year Revenue', marker='o', color='#1f77b4', linewidth=2)
    sns.lineplot(x='Month', y='Revenue', data=previous_df, label='Previous Year Revenue', marker='s', color='#ff7f0e', linewidth=2)
    plt.title(f'Revenue Trend - {company_name}', fontsize=14, weight='bold', pad=10)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    revenue_trend_path = 'revenue_trend.png'
    plt.savefig(revenue_trend_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(revenue_trend_path)
    
    # Bar Chart: Profit Comparison
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    months = current_df['Month']
    x = np.arange(len(months))
    plt.bar(x - bar_width/2, current_df['Profit'], bar_width, label='Current Year Profit', color='#2ca02c')
    plt.bar(x + bar_width/2, previous_df['Profit'], bar_width, label='Previous Year Profit', color='#d62728')
    plt.title(f'Profit Comparison - {company_name}', fontsize=14, weight='bold', pad=10)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Profit ($)', fontsize=12)
    plt.xticks(x, months, rotation=45, fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()
    profit_comparison_path = 'profit_comparison.png'
    plt.savefig(profit_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(profit_comparison_path)
    
    # Pie Chart: Expense Breakdown
    plt.figure(figsize=(8, 8))
    expenses = current_df['Expenses'].sum()
    other = current_df['Revenue'].sum() - expenses - current_df['Profit'].sum()
    plt.pie([expenses, other], labels=['Expenses', 'Other'], autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'], textprops={'fontsize': 12})
    plt.title(f'Expense Breakdown (Current Year) - {company_name}', fontsize=14, weight='bold', pad=10)
    plt.tight_layout()
    expense_breakdown_path = 'expense_breakdown.png'
    plt.savefig(expense_breakdown_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(expense_breakdown_path)
    
    # Area Chart: Revenue vs Expenses
    plt.figure(figsize=(10, 6))
    plt.stackplot(current_df['Month'], current_df['Revenue'], current_df['Expenses'], labels=['Revenue', 'Expenses'], colors=['#1f77b4', '#ff7f0e'])
    plt.title(f'Revenue vs Expenses (Current Year) - {company_name}', fontsize=14, weight='bold', pad=10)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Amount ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    area_chart_path = 'area_chart.png'
    plt.savefig(area_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(area_chart_path)
    
    # Box Plot: Profit Distribution
    plt.figure(figsize=(10, 6))
    profit_data = pd.DataFrame({
        'Current Year': current_df['Profit'],
        'Previous Year': previous_df['Profit']
    })
    sns.boxplot(data=profit_data, palette=['#2ca02c', '#d62728'])
    plt.title(f'Profit Distribution - {company_name}', fontsize=14, weight='bold', pad=10)
    plt.ylabel('Profit ($)', fontsize=12)
    plt.tight_layout()
    box_plot_path = 'box_plot.png'
    plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(box_plot_path)
    
    # # Heatmap: Correlation Matrix
    # plt.figure(figsize=(8, 6))
    # corr_data = current_df[['Revenue', 'Expenses', 'Profit']].corr()
    # sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f', annot_kws={'fontsize': 12})
    # plt.title(f'Correlation Matrix (Current Year) - {company_name}', fontsize=14, weight='bold', pad=10)
    # plt.tight_layout()
    # heatmap_path = 'heatmap.png'
    # plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    # plt.close()
    # image_paths.append(heatmap_path)
    
    # Table Chart: Monthly Financial Details
    plt.figure(figsize=(12, 8))
    table_data = current_df[['Month', 'Revenue', 'Expenses', 'Profit', 'Profit Margin']].copy()
    table_data['Revenue'] = table_data['Revenue'].apply(lambda x: f"${x:,.2f}")
    table_data['Expenses'] = table_data['Expenses'].apply(lambda x: f"${x:,.2f}")
    table_data['Profit'] = table_data['Profit'].apply(lambda x: f"${x:,.2f}")
    table_data['Profit Margin'] = table_data['Profit Margin'].apply(lambda x: f"{x*100:.2f}%")
    table = plt.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.axis('off')
    plt.title(f'Monthly Financial Details (Current Year) - {company_name}', fontsize=14, weight='bold', pad=10)
    plt.tight_layout()
    table_chart_path = 'table_chart.png'
    plt.savefig(table_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(table_chart_path)
    
    return image_paths

# Function to generate FinAlyzer-inspired PDF report
def generate_pdf_report(company_name, report_text, current_totals, previous_totals, next_year_revenue, revenue_change, current_df, image_paths, kpi_threshold, output_filename='financial_report.pdf'):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    header_style = ParagraphStyle(name='Header', fontSize=16, textColor=colors.white, alignment=TA_CENTER, spaceAfter=12)
    section_style = ParagraphStyle(name='Section', fontSize=12, textColor=colors.black, spaceAfter=8, fontName='Helvetica-Bold')
    narrative_style = ParagraphStyle(name='Narrative', fontSize=10, textColor=colors.black, spaceAfter=6, alignment=TA_JUSTIFY)
    normal_style = ParagraphStyle(name='Normal', fontSize=10, textColor=colors.black, spaceAfter=6)
    
    # Header
    header_table = Table([[Paragraph(f"Financial Report for {company_name}", header_style)]], colWidths=[7.5*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.darkblue),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 12)
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 12))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", section_style))
    exec_summary = f"This report, powered by FinAlyzer-inspired automation, provides a unified analysis of {company_name}'s financial performance across entities for the current and previous years. It consolidates financial data, delivers actionable insights through KPIs, variance analysis, and predictive forecasts, and supports strategic decision-making with detailed visualizations."
    elements.append(Paragraph(exec_summary, narrative_style))
    elements.append(Spacer(1, 12))
    
    # Financial Consolidation
    elements.append(Paragraph("Financial Consolidation", section_style))
    elements.append(Paragraph(f"Data is consolidated across entities (if provided), automating aggregation of Revenue, Expenses, and Profit. Segment-wise profitability is analyzed by entity and month, with KPI alerts for profit margins below {kpi_threshold*100:.0f}%.", narrative_style))
    
    # Entity-Level Summary
    if 'Entity' in current_df.columns:
        elements.append(Paragraph("Entity-Level Summary (Current Year)", section_style))
        entity_summary = current_df.groupby('Entity')[['Revenue', 'Expenses', 'Profit']].sum()
        entity_data = [['Entity', 'Revenue', 'Expenses', 'Profit', 'Profit Margin']]
        for entity, row in entity_summary.iterrows():
            profit_margin = (row['Profit'] / row['Revenue'] * 100) if row['Revenue'] > 0 else 0
            entity_data.append([entity, f"${row['Revenue']:,.2f}", f"${row['Expenses']:,.2f}", f"${row['Profit']:,.2f}", f"{profit_margin:.2f}%"])
        entity_table = Table(entity_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        entity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(entity_table)
        elements.append(Spacer(1, 12))
    
    # KPI Alerts
    elements.append(Paragraph("KPI Alerts", section_style))
    current_df['Profit Margin'] = current_df['Profit'] / current_df['Revenue']
    kpi_alerts = current_df[current_df['Profit Margin'] < kpi_threshold][['Month', 'Profit Margin']]
    kpi_alert_data = [['Month', 'Profit Margin']]
    if not kpi_alerts.empty:
        for _, row in kpi_alerts.iterrows():
            kpi_alert_data.append([row['Month'], f"{row['Profit Margin']*100:.2f}%"])
    else:
        kpi_alert_data.append(['None', 'No alerts'])
    kpi_table = Table(kpi_alert_data, colWidths=[3*inch, 2*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))
    
    # Variance Analysis
    if 'Budget' in current_df.columns:
        elements.append(Paragraph("Variance Analysis (Actual vs. Budget)", section_style))
        current_df['Variance'] = current_df['Revenue'] - current_df['Budget']
        variance_data = [['Month', 'Actual', 'Budget', 'Variance']]
        for _, row in current_df.iterrows():
            variance_data.append([row['Month'], f"${row['Revenue']:,.2f}", f"${row['Budget']:,.2f}", f"${row['Variance']:,.2f}"])
        variance_table = Table(variance_data, colWidths=[2*inch, 2*inch, 2*inch, 2*inch])
        variance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(variance_table)
        elements.append(Spacer(1, 12))
    
    # Summary Table
    elements.append(Paragraph("Summary", section_style))
    elements.append(Paragraph(f"This table consolidates {company_name}'s financial performance across all entities.", narrative_style))
    summary_data = [
        ['Metric', 'Current Year', 'Previous Year'],
        ['Revenue', f"${current_totals['Revenue']:,.2f}", f"${previous_totals['Revenue']:,.2f}"],
        ['Expenses', f"${current_totals['Expenses']:,.2f}", f"${previous_totals['Expenses']:,.2f}"],
        ['Profit', f"${current_totals['Profit']:,.2f}", f"${previous_totals['Profit']:,.2f}"]
    ]
    summary_table = Table(summary_data, colWidths=[2*inch, 2.5*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))
    
    # Year-over-Year Comparison
    elements.append(Paragraph("Year-over-Year Comparison", section_style))
    elements.append(Paragraph(f"This section highlights growth trends and operational efficiencies for {company_name}.", narrative_style))
    comparison_data = [['Metric', 'Change (%)']]
    for key in current_totals:
        change = ((current_totals[key] - previous_totals[key]) / previous_totals[key]) * 100
        comparison_data.append([key, f"{change:+.2f}%"])
    comparison_table = Table(comparison_data, colWidths=[3*inch, 2*inch])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(comparison_table)
    elements.append(Spacer(1, 12))
    
    # Predictions
    elements.append(Paragraph("Predictions", section_style))
    elements.append(Paragraph(f"This section provides predictive insights for {company_name}'s future performance.", narrative_style))
    prediction_data = [
        ['Metric', 'Value'],
        ['Predicted Revenue', f"${next_year_revenue:,.2f}"],
        ['Predicted Revenue Change', f"{revenue_change:+.2f}%"]
    ]
    prediction_table = Table(prediction_data, colWidths=[3*inch, 2*inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(prediction_table)
    elements.append(Spacer(1, 12))
    
    # Visualizations
    elements.append(Paragraph("Visualizations", section_style))
    elements.append(Paragraph(f"These visualizations, inspired by FinAlyzer and Power BI, provide insights into {company_name}'s financial performance, covering trends, comparisons, and correlations.", narrative_style))
    elements.append(Spacer(1, 12))
    for i, image_path in enumerate(image_paths):
        img = Image(image_path, width=6*inch, height=4*inch)
        elements.append(img)
        elements.append(Spacer(1, 12))
        if (i + 1) % 2 == 0 and i < len(image_paths) - 1:
            elements.append(PageBreak())
    
    # Audit Trail
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph("Audit Trail", section_style))
    elements.append(Paragraph(f"Data uploaded and processed on: {timestamp}", normal_style))
    
    # Build PDF with custom canvas for page numbers
    def add_page_numbers(canvas, doc):
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(7.5*inch, 0.3*inch, text)
    
    doc.build(elements, onFirstPage=add_page_numbers, onLaterPages=add_page_numbers)
    
    # Clean up image files
    for image_path in image_paths:
        if os.path.exists(image_path):
            os.remove(image_path)
    
    buffer.seek(0)
    return buffer

# Streamlit UI
st.title("Financial Report Generator")
st.write("Upload financial data (CSV) to generate a FinAlyzer-inspired report with Power BI visuals and traditional elements.")

company_name = st.text_input("Enter Company Name", "Example Inc.")
kpi_threshold = st.slider("Set Profit Margin Threshold for KPI Alerts (%)", 0, 100, 30) / 100
current_year_file = st.file_uploader("Upload Current Year Financial Data (CSV)", type="csv")
previous_year_file = st.file_uploader("Upload Previous Year Financial Data (CSV)", type="csv")

if current_year_file and previous_year_file:
    current_df = load_data(current_year_file, "Current Year")
    previous_df = load_data(previous_year_file, "Previous Year")
    
    if current_df is not None and previous_df is not None:
        # Generate report text and metrics
        report_content, current_totals, previous_totals, next_year_revenue, revenue_change, current_df = generate_report_text(current_df, previous_df, company_name, kpi_threshold)
        
        # Display report
        st.subheader("Generated Report")
        st.text_area("Report Content", report_content, height=400)
        
        # Generate visualizations
        st.subheader("Visualizations")
        image_paths = generate_visualizations(current_df, previous_df, company_name, kpi_threshold)
        for image_path in image_paths:
            st.image(image_path)
        
        # Generate and download PDF
        pdf_buffer = generate_pdf_report(company_name, report_content, current_totals, previous_totals, next_year_revenue, revenue_change, current_df, image_paths, kpi_threshold)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_buffer,
            file_name=f"{company_name}_financial_report.pdf",
            mime="application/pdf"
        )
else:
    st.info("Please upload both CSV files to generate the report.")














# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# import numpy as np
# from io import StringIO, BytesIO
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
# from reportlab.lib import colors
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch
# from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
# from reportlab.pdfgen import canvas
# import os
# from datetime import datetime

# # Streamlit app configuration - must be first command
# st.set_page_config(page_title="Financial Report Generator", layout="wide")

# # Custom CSS for enhanced styling
# st.markdown("""
# <style>
#     .header {
#         font-size: 36px !important;
#         font-weight: bold !important;
#         color: #2c3e50 !important;
#         text-align: center;
#         padding: 20px;
#         background: linear-gradient(90deg, #f8f9fa, #e9ecef, #f8f9fa);
#         border-radius: 10px;
#         margin-bottom: 30px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
#     .subheader {
#         font-size: 18px !important;
#         color: #495057 !important;
#         text-align: center;
#         margin-bottom: 30px;
#     }
#     .stButton>button {
#         background-color: #4CAF50 !important;
#         color: white !important;
#         font-weight: bold !important;
#         padding: 10px 24px !important;
#         border-radius: 8px !important;
#         transition: all 0.3s ease;
#     }
#     .stButton>button:hover {
#         background-color: #45a049 !important;
#         transform: scale(1.02);
#     }
#     .stTextInput>div>div>input, .stSlider>div>div>div>div {
#         border-radius: 8px !important;
#         border: 1px solid #ced4da !important;
#         padding: 8px !important;
#     }
#     .stFileUploader>div>div {
#         border-radius: 8px !important;
#         border: 1px solid #ced4da !important;
#         padding: 20px !important;
#     }
#     .stExpander>div>div>div>div {
#         background-color: #f8f9fa !important;
#         border-radius: 8px !important;
#     }
#     .stTabs>div>div>div>div {
#         border-radius: 8px 8px 0 0 !important;
#     }
#     .css-1aumxhk {
#         background-color: #f8f9fa !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Main header with enhanced styling
# st.markdown('<div class="header">Financial Report Generator</div>', unsafe_allow_html=True)
# st.markdown('<div class="subheader">Generate comprehensive financial reports with automated analysis and professional visualizations</div>', unsafe_allow_html=True)

# # Function to load and validate CSV data
# def load_data(file, year):
#     try:
#         df = pd.read_csv(file)
#         required_columns = ['Month', 'Revenue', 'Expenses', 'Profit']
#         if not all(col in df.columns for col in required_columns):
#             st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error loading file: {e}")
#         return None

# # Function to generate financial report text and metrics
# def generate_report_text(current_df, previous_df, company_name, kpi_threshold):
#     report = StringIO()
    
#     # Executive Summary
#     report.write(f"EXECUTIVE SUMMARY\n{'='*50}\n")
#     report.write(f"This automated financial report provides a comprehensive analysis of {company_name}'s financial performance comparing the current and previous years. The report includes:\n")
#     report.write("- Consolidated financial metrics across all business units\n")
#     report.write("- Key performance indicators with threshold-based alerts\n")
#     report.write("- Detailed variance analysis against budgets (when available)\n")
#     report.write("- Year-over-year comparison of financial performance\n")
#     report.write("- Predictive analytics for future performance\n\n")
    
#     # Calculate key metrics
#     current_totals = {
#         'Revenue': current_df['Revenue'].sum(),
#         'Expenses': current_df['Expenses'].sum(),
#         'Profit': current_df['Profit'].sum()
#     }
#     previous_totals = {
#         'Revenue': previous_df['Revenue'].sum(),
#         'Expenses': previous_df['Expenses'].sum(),
#         'Profit': previous_df['Profit'].sum()
#     }
    
#     # Entity-Level Analysis
#     entity_analysis = ""
#     if 'Entity' in current_df.columns:
#         entity_summary = current_df.groupby('Entity')[['Revenue', 'Expenses', 'Profit']].sum()
#         entity_analysis += "\nENTITY-LEVEL PERFORMANCE (CURRENT YEAR)\n" + "-"*50 + "\n"
#         for entity, row in entity_summary.iterrows():
#             profit_margin = (row['Profit'] / row['Revenue'] * 100) if row['Revenue'] > 0 else 0
#             entity_analysis += f"{entity.upper()}:\n"
#             entity_analysis += f"  • Revenue: ${row['Revenue']:,.2f}\n"
#             entity_analysis += f"  • Expenses: ${row['Expenses']:,.2f}\n"
#             entity_analysis += f"  • Profit: ${row['Profit']:,.2f}\n"
#             entity_analysis += f"  • Profit Margin: {profit_margin:.2f}%\n\n"
    
#     # KPI Alerts
#     current_df['Profit Margin'] = current_df['Profit'] / current_df['Revenue']
#     kpi_alerts = current_df[current_df['Profit Margin'] < kpi_threshold][['Month', 'Profit Margin']]
#     kpi_alert_text = f"\nKPI ALERTS (PROFIT MARGIN < {kpi_threshold*100:.0f}%)\n" + "-"*50 + "\n"
#     if not kpi_alerts.empty:
#         for _, row in kpi_alerts.iterrows():
#             kpi_alert_text += f"- {row['Month']}: Profit Margin {row['Profit Margin']*100:.2f}%\n"
#     else:
#         kpi_alert_text += "- No months below the profit margin threshold.\n"
    
#     # Variance Analysis
#     variance_analysis = ""
#     if 'Budget' in current_df.columns:
#         current_df['Variance'] = current_df['Revenue'] - current_df['Budget']
#         variance_analysis += f"\nVARIANCE ANALYSIS (ACTUAL VS BUDGET)\n" + "-"*50 + "\n"
#         for _, row in current_df.iterrows():
#             variance_pct = (row['Variance'] / row['Budget'] * 100) if row['Budget'] != 0 else 0
#             variance_analysis += f"{row['Month']}:\n"
#             variance_analysis += f"  • Actual Revenue: ${row['Revenue']:,.2f}\n"
#             variance_analysis += f"  • Budget: ${row['Budget']:,.2f}\n"
#             variance_analysis += f"  • Variance: ${row['Variance']:,.2f} ({variance_pct:+.2f}%)\n\n"
    
#     # Financial Summary
#     report.write(f"\nFINANCIAL SUMMARY\n{'='*50}\n")
#     report.write("CURRENT YEAR TOTALS:\n")
#     for key, value in current_totals.items():
#         report.write(f"  • {key}: ${value:,.2f}\n")
    
#     report.write("\nPREVIOUS YEAR TOTALS:\n")
#     for key, value in previous_totals.items():
#         report.write(f"  • {key}: ${value:,.2f}\n")
    
#     report.write(entity_analysis)
#     report.write(kpi_alert_text)
#     report.write(variance_analysis)
    
#     # Year-over-Year Comparison
#     report.write(f"\nYEAR-OVER-YEAR COMPARISON\n{'='*50}\n")
#     for key in current_totals:
#         change = ((current_totals[key] - previous_totals[key]) / previous_totals[key]) * 100
#         report.write(f"  • {key}: {change:+.2f}% change\n")
    
#     # Predictions
#     report.write(f"\nFINANCIAL FORECAST\n{'='*50}\n")
#     X = np.array([1, 2]).reshape(-1, 1)  # Previous year = 1, Current year = 2
#     y = np.array([previous_totals['Revenue'], current_totals['Revenue']])
#     model = LinearRegression()
#     model.fit(X, y)
#     next_year_revenue = model.predict(np.array([[3]]))[0]
#     revenue_change = ((next_year_revenue - current_totals['Revenue']) / current_totals['Revenue']) * 100
#     report.write(f"  • Predicted Next Year Revenue: ${next_year_revenue:,.2f}\n")
#     report.write(f"  • Projected Growth Rate: {revenue_change:+.2f}%\n")
    
#     # Audit Trail
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     report.write(f"\nAUDIT TRAIL\n{'='*50}\n")
#     report.write(f"Report generated on: {timestamp}\n")
    
#     return report.getvalue(), current_totals, previous_totals, next_year_revenue, revenue_change, current_df

# # Function to generate visualizations
# def generate_visualizations(current_df, previous_df, company_name, kpi_threshold):
#     sns.set_style("whitegrid")
#     plt.rcParams['font.size'] = 12
#     plt.rcParams['axes.labelsize'] = 12
#     plt.rcParams['axes.titlesize'] = 14
#     plt.rcParams['xtick.labelsize'] = 10
#     plt.rcParams['ytick.labelsize'] = 10
#     image_paths = []
    
#     # Visualization 1: Revenue Trend Comparison
#     plt.figure(figsize=(12, 6))
#     ax = sns.lineplot(x='Month', y='Revenue', data=current_df, 
#                      label='Current Year', marker='o', linewidth=2.5, color='#1f77b4')
#     sns.lineplot(x='Month', y='Revenue', data=previous_df, 
#                 label='Previous Year', marker='s', linewidth=2.5, color='#ff7f0e')
#     plt.title(f'Revenue Trend Comparison\n{company_name}', fontweight='bold', pad=20)
#     plt.xlabel('Month', fontweight='bold')
#     plt.ylabel('Revenue ($)', fontweight='bold')
#     plt.legend(title='Fiscal Year')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     revenue_trend_path = 'revenue_trend.png'
#     plt.savefig(revenue_trend_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(revenue_trend_path)
    
#     # Visualization 2: Profit Comparison
#     plt.figure(figsize=(12, 6))
#     bar_width = 0.35
#     months = current_df['Month']
#     x = np.arange(len(months))
#     plt.bar(x - bar_width/2, current_df['Profit'], bar_width, 
#             label='Current Year', color='#2ca02c', edgecolor='white')
#     plt.bar(x + bar_width/2, previous_df['Profit'], bar_width, 
#             label='Previous Year', color='#d62728', edgecolor='white')
#     plt.title(f'Monthly Profit Comparison\n{company_name}', fontweight='bold', pad=20)
#     plt.xlabel('Month', fontweight='bold')
#     plt.ylabel('Profit ($)', fontweight='bold')
#     plt.xticks(x, months, rotation=45)
#     plt.legend(title='Fiscal Year')
#     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     profit_comparison_path = 'profit_comparison.png'
#     plt.savefig(profit_comparison_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(profit_comparison_path)
    
#     # Visualization 3: Expense Breakdown
#     plt.figure(figsize=(8, 8))
#     expenses = current_df['Expenses'].sum()
#     other = current_df['Revenue'].sum() - expenses - current_df['Profit'].sum()
#     plt.pie([expenses, other], labels=['Operating Expenses', 'Other Costs'], 
#             autopct='%1.1f%%', startangle=140, 
#             colors=['#ff9999', '#66b3ff'], textprops={'fontsize': 12},
#             wedgeprops={'edgecolor': 'white', 'linewidth': 1})
#     plt.title(f'Expense Breakdown (Current Year)\n{company_name}', fontweight='bold', pad=20)
#     plt.tight_layout()
#     expense_breakdown_path = 'expense_breakdown.png'
#     plt.savefig(expense_breakdown_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(expense_breakdown_path)
    
#     # Visualization 4: Revenue vs Expenses
#     plt.figure(figsize=(12, 6))
#     plt.stackplot(current_df['Month'], current_df['Revenue'], current_df['Expenses'], 
#                  labels=['Revenue', 'Expenses'], colors=['#1f77b4', '#ff7f0e'], alpha=0.8)
#     plt.title(f'Revenue vs Expenses (Current Year)\n{company_name}', fontweight='bold', pad=20)
#     plt.xlabel('Month', fontweight='bold')
#     plt.ylabel('Amount ($)', fontweight='bold')
#     plt.legend(loc='upper left', fontsize=10)
#     plt.xticks(rotation=45)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     area_chart_path = 'area_chart.png'
#     plt.savefig(area_chart_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(area_chart_path)
    
#     # Visualization 5: Profit Distribution
#     plt.figure(figsize=(10, 6))
#     profit_data = pd.DataFrame({
#         'Current Year': current_df['Profit'],
#         'Previous Year': previous_df['Profit']
#     })
#     sns.boxplot(data=profit_data, palette=['#2ca02c', '#d62728'], width=0.5)
#     plt.title(f'Profit Distribution Comparison\n{company_name}', fontweight='bold', pad=20)
#     plt.ylabel('Profit ($)', fontweight='bold')
#     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     box_plot_path = 'box_plot.png'
#     plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(box_plot_path)
    
#     # Visualization 6: Financial Metrics Table
#     plt.figure(figsize=(12, 8))
#     table_data = current_df[['Month', 'Revenue', 'Expenses', 'Profit']].copy()
#     table_data['Revenue'] = table_data['Revenue'].apply(lambda x: f"${x:,.2f}")
#     table_data['Expenses'] = table_data['Expenses'].apply(lambda x: f"${x:,.2f}")
#     table_data['Profit'] = table_data['Profit'].apply(lambda x: f"${x:,.2f}")
#     table_data['Profit Margin'] = (current_df['Profit'] / current_df['Revenue'] * 100).apply(lambda x: f"{x:.2f}%")
    
#     col_widths = [1.5, 2, 2, 2, 2]
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.axis('tight')
#     ax.axis('off')
#     table = ax.table(cellText=table_data.values, 
#                     colLabels=table_data.columns, 
#                     cellLoc='center', 
#                     loc='center',
#                     colWidths=[w/(sum(col_widths)) for w in col_widths])
    
#     table.auto_set_font_size(False)
#     table.set_fontsize(11)
#     table.scale(1.2, 1.5)
    
#     for (i, j), cell in table.get_celld().items():
#         if i == 0:
#             cell.set_text_props(fontweight='bold')
#             cell.set_facecolor('#2c3e50')
#             cell.set_text_props(color='white')
#         else:
#             cell.set_facecolor('#f8f9fa' if i%2==0 else '#e9ecef')
    
#     plt.title(f'Monthly Financial Details (Current Year)\n{company_name}', 
#              fontweight='bold', pad=20, y=1.05)
#     plt.tight_layout()
#     table_chart_path = 'table_chart.png'
#     plt.savefig(table_chart_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     image_paths.append(table_chart_path)
    
#     return image_paths

# # Function to generate PDF report
# def generate_pdf_report(company_name, report_text, current_totals, previous_totals, 
#                       next_year_revenue, revenue_change, current_df, image_paths, 
#                       kpi_threshold, output_filename='financial_report.pdf'):
#     buffer = BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=letter, 
#                            topMargin=0.5*inch, bottomMargin=0.5*inch,
#                            rightMargin=0.5*inch, leftMargin=0.5*inch)
#     elements = []
#     styles = getSampleStyleSheet()
    
#     # Custom styles
#     title_style = ParagraphStyle(
#         name='Title',
#         parent=styles['Heading1'],
#         fontSize=18,
#         leading=22,
#         textColor=colors.HexColor('#2c3e50'),
#         alignment=TA_CENTER,
#         spaceAfter=12,
#         fontName='Helvetica-Bold'
#     )
    
#     section_style = ParagraphStyle(
#         name='Section',
#         parent=styles['Heading2'],
#         fontSize=14,
#         leading=18,
#         textColor=colors.HexColor('#2c3e50'),
#         spaceAfter=8,
#         fontName='Helvetica-Bold'
#     )
    
#     body_style = ParagraphStyle(
#         name='Body',
#         parent=styles['BodyText'],
#         fontSize=10,
#         leading=14,
#         textColor=colors.black,
#         spaceAfter=6,
#         alignment=TA_JUSTIFY
#     )
    
#     # Cover Page
#     elements.append(Spacer(1, 2*inch))
#     elements.append(Paragraph(f"Financial Analysis Report", title_style))
#     elements.append(Spacer(1, 0.5*inch))
#     elements.append(Paragraph(f"Prepared for: {company_name}", styles['Heading2']))
#     elements.append(Spacer(1, 0.25*inch))
#     elements.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", styles['BodyText']))
#     elements.append(PageBreak())
    
#     # Table of Contents
#     elements.append(Paragraph("Table of Contents", section_style))
#     toc = [
#         ("1. Executive Summary", "2"),
#         ("2. Financial Overview", "2"),
#         ("3. Performance Analysis", "3"),
#         ("4. Visualizations", "4"),
#         ("5. Forecast & Recommendations", str(len(image_paths)+4))
#     ]
    
#     toc_table = Table(toc, colWidths=[4*inch, 1*inch])
#     toc_table.setStyle(TableStyle([
#         ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
#         ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.grey),
#         ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
#     ]))
#     elements.append(toc_table)
#     elements.append(PageBreak())
    
#     # Executive Summary
#     elements.append(Paragraph("1. Executive Summary", section_style))
#     summary_text = f"""
#     This report provides a comprehensive analysis of {company_name}'s financial performance, comparing the current fiscal year with the previous year. 
#     The analysis includes detailed examination of revenue trends, expense patterns, profitability metrics, and predictive insights for future performance.
#     """
#     elements.append(Paragraph(summary_text, body_style))
#     elements.append(Spacer(1, 0.25*inch))
    
#     # Key Findings
#     elements.append(Paragraph("Key Findings", styles['Heading3']))
#     findings = [
#         f"• Total Revenue: ${current_totals['Revenue']:,.2f} ({((current_totals['Revenue']-previous_totals['Revenue'])/previous_totals['Revenue'])*100:+.2f}% YoY)",
#         f"• Total Profit: ${current_totals['Profit']:,.2f} ({((current_totals['Profit']-previous_totals['Profit'])/previous_totals['Profit'])*100:+.2f}% YoY)",
#         f"• Predicted Next Year Revenue: ${next_year_revenue:,.2f} ({revenue_change:+.2f}% growth)",
#         f"• KPI Threshold: Profit Margin < {kpi_threshold*100:.0f}%"
#     ]
#     for finding in findings:
#         elements.append(Paragraph(finding, body_style))
#     elements.append(PageBreak())
    
#     # Financial Overview
#     elements.append(Paragraph("2. Financial Overview", section_style))
    
#     # Summary Table
#     summary_data = [
#         ['Metric', 'Current Year', 'Previous Year', 'Change (%)'],
#         ['Revenue', f"${current_totals['Revenue']:,.2f}", f"${previous_totals['Revenue']:,.2f}", 
#          f"{((current_totals['Revenue']-previous_totals['Revenue'])/previous_totals['Revenue'])*100:+.2f}%"],
#         ['Expenses', f"${current_totals['Expenses']:,.2f}", f"${previous_totals['Expenses']:,.2f}", 
#          f"{((current_totals['Expenses']-previous_totals['Expenses'])/previous_totals['Expenses'])*100:+.2f}%"],
#         ['Profit', f"${current_totals['Profit']:,.2f}", f"${previous_totals['Profit']:,.2f}", 
#          f"{((current_totals['Profit']-previous_totals['Profit'])/previous_totals['Profit'])*100:+.2f}%"]
#     ]
    
#     summary_table = Table(summary_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1*inch])
#     summary_table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, 0), 10),
#         ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
#         ('GRID', (0, 0), (-1, -1), 1, colors.grey)
#     ]))
#     elements.append(summary_table)
#     elements.append(Spacer(1, 0.25*inch))
    
#     # Performance Analysis
#     elements.append(Paragraph("3. Performance Analysis", section_style))
    
#     # Add visualizations to PDF
#     elements.append(Paragraph("4. Visualizations", section_style))
#     for i, image_path in enumerate(image_paths):
#         img = Image(image_path, width=6*inch, height=4*inch)
#         elements.append(img)
#         elements.append(Spacer(1, 0.25*inch))
#         if (i+1) % 2 == 0 and i < len(image_paths)-1:
#             elements.append(PageBreak())
    
#     # Forecast & Recommendations
#     elements.append(PageBreak())
#     elements.append(Paragraph("5. Forecast & Recommendations", section_style))
#     forecast_text = f"""
#     Based on the linear regression analysis of historical revenue data, the projected revenue for the next fiscal year is estimated at ${next_year_revenue:,.2f}, 
#     representing a {revenue_change:+.2f}% change from the current year. This forecast suggests {"growth" if revenue_change > 0 else "decline"} in the company's financial performance.
#     """
#     elements.append(Paragraph(forecast_text, body_style))
    
#     elements.append(Paragraph("Recommendations:", styles['Heading3']))
#     recommendations = [
#         "• Continue current growth strategies if revenue projection is positive",
#         "• Review expense management if profit margins are below threshold",
#         "• Investigate months with significant budget variances",
#         "• Consider diversifying revenue streams if growth is stagnant"
#     ]
#     for rec in recommendations:
#         elements.append(Paragraph(rec, body_style))
    
#     # Footer with page numbers
#     def add_page_numbers(canvas, doc):
#         page_num = canvas.getPageNumber()
#         text = f"Page {page_num}"
#         canvas.setFont("Helvetica", 9)
#         canvas.drawRightString(7.5*inch, 0.5*inch, text)
    
#     doc.build(elements, onFirstPage=add_page_numbers, onLaterPages=add_page_numbers)
    
#     # Clean up image files
#     for image_path in image_paths:
#         if os.path.exists(image_path):
#             os.remove(image_path)
    
#     buffer.seek(0)
#     return buffer

# # Main application UI
# col1, col2 = st.columns(2)

# with col1:
#     company_name = st.text_input("Enter Company Name", "Example Inc.", key="company_name")
#     kpi_threshold = st.slider(
#         "Set Profit Margin Threshold for KPI Alerts (%)", 
#         0, 100, 30,
#         help="Months with profit margins below this threshold will trigger alerts"
#     ) / 100

# with col2:
#     current_year_file = st.file_uploader(
#         "Upload Current Year Financial Data (CSV)", 
#         type="csv",
#         help="CSV should contain columns: Month, Revenue, Expenses, Profit"
#     )
#     previous_year_file = st.file_uploader(
#         "Upload Previous Year Financial Data (CSV)", 
#         type="csv",
#         help="CSV should contain same structure as current year data"
#     )

# if current_year_file and previous_year_file:
#     current_df = load_data(current_year_file, "Current Year")
#     previous_df = load_data(previous_year_file, "Previous Year")
    
#     if current_df is not None and previous_df is not None:
#         # Generate report content
#         with st.spinner('Generating financial analysis...'):
#             report_content, current_totals, previous_totals, next_year_revenue, revenue_change, current_df = generate_report_text(
#                 current_df, previous_df, company_name, kpi_threshold
#             )
            
#             # Generate visualizations
#             image_paths = generate_visualizations(current_df, previous_df, company_name, kpi_threshold)
            
#             # Generate PDF report
#             pdf_buffer = generate_pdf_report(
#                 company_name, report_content, current_totals, previous_totals, 
#                 next_year_revenue, revenue_change, current_df, image_paths, kpi_threshold
#             )
        
#         # Display results
#         st.success('Analysis complete!')
        
#         # Report section
#         with st.expander("📄 View Detailed Report Analysis", expanded=True):
#             st.subheader(f"{company_name} - Financial Performance Analysis")
#             st.text_area("Full Report Details", report_content, height=400)
        
#         # Visualizations section
#         st.markdown("---")
#         st.subheader("📊 Financial Visualizations")
        
#         # Display charts in tabs
#         tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#             "Revenue Trends", "Profit Comparison", "Expense Breakdown", 
#             "Revenue vs Expenses", "Profit Distribution", "Monthly Details"
#         ])
        
#         with tab1:
#             st.image(image_paths[0], use_column_width=True, caption="Monthly Revenue Trends Comparison")
#         with tab2:
#             st.image(image_paths[1], use_column_width=True, caption="Monthly Profit Comparison")
#         with tab3:
#             st.image(image_paths[2], use_column_width=True, caption="Current Year Expense Breakdown")
#         with tab4:
#             st.image(image_paths[3], use_column_width=True, caption="Revenue vs Expenses (Current Year)")
#         with tab5:
#             st.image(image_paths[4], use_column_width=True, caption="Profit Distribution Comparison")
#         with tab6:
#             st.image(image_paths[5], use_column_width=True, caption="Monthly Financial Details")
        
#         # Download section
#         st.markdown("---")
#         st.subheader("📥 Download Full Report")
        
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             st.download_button(
#                 label="Download PDF Report",
#                 data=pdf_buffer,
#                 file_name=f"{company_name.replace(' ', '_')}_Financial_Report.pdf",
#                 mime="application/pdf",
#                 help="Generate a professional PDF report with all analysis and visualizations"
#             )
# else:
#     st.warning("⚠️ Please upload both current and previous year financial data to generate the report")













# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# import numpy as np
# from io import StringIO, BytesIO
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
# from reportlab.lib import colors
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch
# from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
# from reportlab.pdfgen import canvas
# import os
# import tempfile
# from datetime import datetime

# # Streamlit app configuration - must be first command
# st.set_page_config(page_title="Financial Report Generator", layout="wide")

# # Custom CSS for enhanced styling
# st.markdown("""
# <style>
#     .header {
#         font-size: 36px !important;
#         font-weight: bold !important;
#         color: #2c3e50 !important;
#         text-align: center;
#         padding: 20px;
#         background: linear-gradient(90deg, #f8f9fa, #e9ecef, #f8f9fa);
#         border-radius: 10px;
#         margin-bottom: 30px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
#     .subheader {
#         font-size: 18px !important;
#         color: #495057 !important;
#         text-align: center;
#         margin-bottom: 30px;
#     }
#     .stButton>button {
#         background-color: #4CAF50 !important;
#         color: white !important;
#         font-weight: bold !important;
#         padding: 10px 24px !important;
#         border-radius: 8px !important;
#         transition: all 0.3s ease;
#     }
#     .stButton>button:hover {
#         background-color: #45a049 !important;
#         transform: scale(1.02);
#     }
#     .stTextInput>div>div>input, .stSlider>div>div>div>div {
#         border-radius: 8px !important;
#         border: 1px solid #ced4da !important;
#         padding: 8px !important;
#     }
#     .stFileUploader>div>div {
#         border-radius: 8px !important;
#         border: 1px solid #ced4da !important;
#         padding: 20px !important;
#     }
#     .stExpander>div>div>div>div {
#         background-color: #f8f9fa !important;
#         border-radius: 8px !important;
#     }
#     .stTabs>div>div>div>div {
#         border-radius: 8px 8px 0 0 !important;
#     }
#     .css-1aumxhk {
#         background-color: #f8f9fa !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Main header with enhanced styling
# st.markdown('<div class="header">Financial Report Generator</div>', unsafe_allow_html=True)
# st.markdown('<div class="subheader">Generate comprehensive financial reports with automated analysis and professional visualizations</div>', unsafe_allow_html=True)

# # Function to load and validate CSV data
# def load_data(file, year):
#     try:
#         df = pd.read_csv(file)
#         required_columns = ['Month', 'Revenue', 'Expenses', 'Profit']
#         if not all(col in df.columns for col in required_columns):
#             st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error loading file: {e}")
#         return None

# # Function to generate financial report text and metrics
# def generate_report_text(current_df, previous_df, company_name, kpi_threshold):
#     report = StringIO()
    
#     # Executive Summary
#     report.write(f"EXECUTIVE SUMMARY\n{'='*50}\n")
#     report.write(f"This automated financial report provides a comprehensive analysis of {company_name}'s financial performance comparing the current and previous years. The report includes:\n")
#     report.write("- Consolidated financial metrics across all business units\n")
#     report.write("- Key performance indicators with threshold-based alerts\n")
#     report.write("- Detailed variance analysis against budgets (when available)\n")
#     report.write("- Year-over-year comparison of financial performance\n")
#     report.write("- Predictive analytics for future performance\n\n")
    
#     # Calculate key metrics
#     current_totals = {
#         'Revenue': current_df['Revenue'].sum(),
#         'Expenses': current_df['Expenses'].sum(),
#         'Profit': current_df['Profit'].sum()
#     }
#     previous_totals = {
#         'Revenue': previous_df['Revenue'].sum(),
#         'Expenses': previous_df['Expenses'].sum(),
#         'Profit': previous_df['Profit'].sum()
#     }
    
#     # Entity-Level Analysis
#     entity_analysis = ""
#     if 'Entity' in current_df.columns:
#         entity_summary = current_df.groupby('Entity')[['Revenue', 'Expenses', 'Profit']].sum()
#         entity_analysis += "\nENTITY-LEVEL PERFORMANCE (CURRENT YEAR)\n" + "-"*50 + "\n"
#         for entity, row in entity_summary.iterrows():
#             profit_margin = (row['Profit'] / row['Revenue'] * 100) if row['Revenue'] > 0 else 0
#             entity_analysis += f"{entity.upper()}:\n"
#             entity_analysis += f"  • Revenue: ${row['Revenue']:,.2f}\n"
#             entity_analysis += f"  • Expenses: ${row['Expenses']:,.2f}\n"
#             entity_analysis += f"  • Profit: ${row['Profit']:,.2f}\n"
#             entity_analysis += f"  • Profit Margin: {profit_margin:.2f}%\n\n"
    
#     # KPI Alerts
#     current_df['Profit Margin'] = current_df['Profit'] / current_df['Revenue']
#     kpi_alerts = current_df[current_df['Profit Margin'] < kpi_threshold][['Month', 'Profit Margin']]
#     kpi_alert_text = f"\nKPI ALERTS (PROFIT MARGIN < {kpi_threshold*100:.0f}%)\n" + "-"*50 + "\n"
#     if not kpi_alerts.empty:
#         for _, row in kpi_alerts.iterrows():
#             kpi_alert_text += f"- {row['Month']}: Profit Margin {row['Profit Margin']*100:.2f}%\n"
#     else:
#         kpi_alert_text += "- No months below the profit margin threshold.\n"
    
#     # Variance Analysis
#     variance_analysis = ""
#     if 'Budget' in current_df.columns:
#         current_df['Variance'] = current_df['Revenue'] - current_df['Budget']
#         variance_analysis += f"\nVARIANCE ANALYSIS (ACTUAL VS BUDGET)\n" + "-"*50 + "\n"
#         for _, row in current_df.iterrows():
#             variance_pct = (row['Variance'] / row['Budget'] * 100) if row['Budget'] != 0 else 0
#             variance_analysis += f"{row['Month']}:\n"
#             variance_analysis += f"  • Actual Revenue: ${row['Revenue']:,.2f}\n"
#             variance_analysis += f"  • Budget: ${row['Budget']:,.2f}\n"
#             variance_analysis += f"  • Variance: ${row['Variance']:,.2f} ({variance_pct:+.2f}%)\n\n"
    
#     # Financial Summary
#     report.write(f"\nFINANCIAL SUMMARY\n{'='*50}\n")
#     report.write("CURRENT YEAR TOTALS:\n")
#     for key, value in current_totals.items():
#         report.write(f"  • {key}: ${value:,.2f}\n")
    
#     report.write("\nPREVIOUS YEAR TOTALS:\n")
#     for key, value in previous_totals.items():
#         report.write(f"  • {key}: ${value:,.2f}\n")
    
#     report.write(entity_analysis)
#     report.write(kpi_alert_text)
#     report.write(variance_analysis)
    
#     # Year-over-Year Comparison
#     report.write(f"\nYEAR-OVER-YEAR COMPARISON\n{'='*50}\n")
#     for key in current_totals:
#         change = ((current_totals[key] - previous_totals[key]) / previous_totals[key]) * 100
#         report.write(f"  • {key}: {change:+.2f}% change\n")
    
#     # Predictions
#     report.write(f"\nFINANCIAL FORECAST\n{'='*50}\n")
#     X = np.array([1, 2]).reshape(-1, 1)  # Previous year = 1, Current year = 2
#     y = np.array([previous_totals['Revenue'], current_totals['Revenue']])
#     model = LinearRegression()
#     model.fit(X, y)
#     next_year_revenue = model.predict(np.array([[3]]))[0]
#     revenue_change = ((next_year_revenue - current_totals['Revenue']) / current_totals['Revenue']) * 100
#     report.write(f"  • Predicted Next Year Revenue: ${next_year_revenue:,.2f}\n")
#     report.write(f"  • Projected Growth Rate: {revenue_change:+.2f}%\n")
    
#     # Audit Trail
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     report.write(f"\nAUDIT TRAIL\n{'='*50}\n")
#     report.write(f"Report generated on: {timestamp}\n")
    
#     return report.getvalue(), current_totals, previous_totals, next_year_revenue, revenue_change, current_df

# # Function to generate visualizations
# def generate_visualizations(current_df, previous_df, company_name, kpi_threshold):
#     temp_dir = tempfile.mkdtemp()
#     sns.set_style("whitegrid")
#     plt.rcParams['font.size'] = 12
#     plt.rcParams['axes.labelsize'] = 12
#     plt.rcParams['axes.titlesize'] = 14
#     plt.rcParams['xtick.labelsize'] = 10
#     plt.rcParams['ytick.labelsize'] = 10
#     image_paths = []
    
#     # Visualization 1: Revenue Trend Comparison
#     plt.figure(figsize=(12, 6))
#     ax = sns.lineplot(x='Month', y='Revenue', data=current_df, 
#                      label='Current Year', marker='o', linewidth=2.5, color='#1f77b4')
#     sns.lineplot(x='Month', y='Revenue', data=previous_df, 
#                 label='Previous Year', marker='s', linewidth=2.5, color='#ff7f0e')
#     plt.title(f'Revenue Trend Comparison\n{company_name}', fontweight='bold', pad=20)
#     plt.xlabel('Month', fontweight='bold')
#     plt.ylabel('Revenue ($)', fontweight='bold')
#     plt.legend(title='Fiscal Year')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     revenue_trend_path = os.path.join(temp_dir, 'revenue_trend.png')
#     try:
#         plt.savefig(revenue_trend_path, dpi=300, bbox_inches='tight')
#         if not os.path.exists(revenue_trend_path):
#             st.error(f"Failed to save {revenue_trend_path}")
#     except Exception as e:
#         st.error(f"Error saving {revenue_trend_path}: {e}")
#     plt.close()
#     image_paths.append(revenue_trend_path)
    
#     # Visualization 2: Profit Comparison
#     plt.figure(figsize=(12, 6))
#     bar_width = 0.35
#     months = current_df['Month']
#     x = np.arange(len(months))
#     plt.bar(x - bar_width/2, current_df['Profit'], bar_width, 
#             label='Current Year', color='#2ca02c', edgecolor='white')
#     plt.bar(x + bar_width/2, previous_df['Profit'], bar_width, 
#             label='Previous Year', color='#d62728', edgecolor='white')
#     plt.title(f'Monthly Profit Comparison\n{company_name}', fontweight='bold', pad=20)
#     plt.xlabel('Month', fontweight='bold')
#     plt.ylabel('Profit ($)', fontweight='bold')
#     plt.xticks(x, months, rotation=45)
#     plt.legend(title='Fiscal Year')
#     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     profit_comparison_path = os.path.join(temp_dir, 'profit_comparison.png')
#     try:
#         plt.savefig(profit_comparison_path, dpi=300, bbox_inches='tight')
#         if not os.path.exists(profit_comparison_path):
#             st.error(f"Failed to save {profit_comparison_path}")
#     except Exception as e:
#         st.error(f"Error saving {profit_comparison_path}: {e}")
#     plt.close()
#     image_paths.append(profit_comparison_path)
    
#     # Visualization 3: Expense Breakdown
#     plt.figure(figsize=(8, 8))
#     expenses = current_df['Expenses'].sum()
#     other = current_df['Revenue'].sum() - expenses - current_df['Profit'].sum()
#     plt.pie([expenses, other], labels=['Operating Expenses', 'Other Costs'], 
#             autopct='%1.1f%%', startangle=140, 
#             colors=['#ff9999', '#66b3ff'], textprops={'fontsize': 12},
#             wedgeprops={'edgecolor': 'white', 'linewidth': 1})
#     plt.title(f'Expense Breakdown (Current Year)\n{company_name}', fontweight='bold', pad=20)
#     plt.tight_layout()
#     expense_breakdown_path = os.path.join(temp_dir, 'expense_breakdown.png')
#     try:
#         plt.savefig(expense_breakdown_path, dpi=300, bbox_inches='tight')
#         if not os.path.exists(expense_breakdown_path):
#             st.error(f"Failed to save {expense_breakdown_path}")
#     except Exception as e:
#         st.error(f"Error saving {expense_breakdown_path}: {e}")
#     plt.close()
#     image_paths.append(expense_breakdown_path)
    
#     # Visualization 4: Revenue vs Expenses
#     plt.figure(figsize=(12, 6))
#     plt.stackplot(current_df['Month'], current_df['Revenue'], current_df['Expenses'], 
#                  labels=['Revenue', 'Expenses'], colors=['#1f77b4', '#ff7f0e'], alpha=0.8)
#     plt.title(f'Revenue vs Expenses (Current Year)\n{company_name}', fontweight='bold', pad=20)
#     plt.xlabel('Month', fontweight='bold')
#     plt.ylabel('Amount ($)', fontweight='bold')
#     plt.legend(loc='upper left', fontsize=10)
#     plt.xticks(rotation=45)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     area_chart_path = os.path.join(temp_dir, 'area_chart.png')
#     try:
#         plt.savefig(area_chart_path, dpi=300, bbox_inches='tight')
#         if not os.path.exists(area_chart_path):
#             st.error(f"Failed to save {area_chart_path}")
#     except Exception as e:
#         st.error(f"Error saving {area_chart_path}: {e}")
#     plt.close()
#     image_paths.append(area_chart_path)
    
#     # Visualization 5: Profit Distribution
#     plt.figure(figsize=(10, 6))
#     profit_data = pd.DataFrame({
#         'Current Year': current_df['Profit'],
#         'Previous Year': previous_df['Profit']
#     })
#     sns.boxplot(data=profit_data, palette=['#2ca02c', '#d62728'], width=0.5)
#     plt.title(f'Profit Distribution Comparison\n{company_name}', fontweight='bold', pad=20)
#     plt.ylabel('Profit ($)', fontweight='bold')
#     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     box_plot_path = os.path.join(temp_dir, 'box_plot.png')
#     try:
#         plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
#         if not os.path.exists(box_plot_path):
#             st.error(f"Failed to save {box_plot_path}")
#     except Exception as e:
#         st.error(f"Error saving {box_plot_path}: {e}")
#     plt.close()
#     image_paths.append(box_plot_path)
    
#     # Visualization 6: Financial Metrics Table
#     plt.figure(figsize=(12, 8))
#     table_data = current_df[['Month', 'Revenue', 'Expenses', 'Profit']].copy()
#     table_data['Revenue'] = table_data['Revenue'].apply(lambda x: f"${x:,.2f}")
#     table_data['Expenses'] = table_data['Expenses'].apply(lambda x: f"${x:,.2f}")
#     table_data['Profit'] = table_data['Profit'].apply(lambda x: f"${x: |,2f}")
#     table_data['Profit Margin'] = (current_df['Profit'] / current_df['Revenue'] * 100).apply(lambda x: f"{x:.2f}%")
    
#     col_widths = [1.5, 2, 2, 2, 2]
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.axis('tight')
#     ax.axis('off')
#     table = ax.table(cellText=table_data.values, 
#                     colLabels=table_data.columns, 
#                     cellLoc='center', 
#                     loc='center',
#                     colWidths=[w/(sum(col_widths)) for w in col_widths])
    
#     table.auto_set_font_size(False)
#     table.set_fontsize(11)
#     table.scale(1.2, 1.5)
    
#     for (i, j), cell in table.get_celld().items():
#         if i == 0:
#             cell.set_text_props(fontweight='bold')
#             cell.set_facecolor('#2c3e50')
#             cell.set_text_props(color='white')
#         else:
#             cell.set_facecolor('#f8f9fa' if i%2==0 else '#e9ecef')
    
#     plt.title(f'Monthly Financial Details (Current Year)\n{company_name}', 
#              fontweight='bold', pad=20, y=1.05)
#     plt.tight_layout()
#     table_chart_path = os.path.join(temp_dir, 'table_chart.png')
#     try:
#         plt.savefig(table_chart_path, dpi=300, bbox_inches='tight')
#         if not os.path.exists(table_chart_path):
#             st.error(f"Failed to save {table_chart_path}")
#     except Exception as e:
#         st.error(f"Error saving {table_chart_path}: {e}")
#     plt.close()
#     image_paths.append(table_chart_path)
    
#     return image_paths, temp_dir

# # Function to generate PDF report
# def generate_pdf_report(company_name, report_text, current_totals, previous_totals, 
#                       next_year_revenue, revenue_change, current_df, image_paths, 
#                       kpi_threshold, output_filename='financial_report.pdf'):
#     buffer = BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=letter, 
#                            topMargin=0.5*inch, bottomMargin=0.5*inch,
#                            rightMargin=0.5*inch, leftMargin=0.5*inch)
#     elements = []
#     styles = getSampleStyleSheet()
    
#     # Custom styles
#     title_style = ParagraphStyle(
#         name='Title',
#         parent=styles['Heading1'],
#         fontSize=18,
#         leading=22,
#         textColor=colors.HexColor('#2c3e50'),
#         alignment=TA_CENTER,
#         spaceAfter=12,
#         fontName='Helvetica-Bold'
#     )
    
#     section_style = ParagraphStyle(
#         name='Section',
#         parent=styles['Heading2'],
#         fontSize=14,
#         leading=18,
#         textColor=colors.HexColor('#2c3e50'),
#         spaceAfter=8,
#         fontName='Helvetica-Bold'
#     )
    
#     body_style = ParagraphStyle(
#         name='Body',
#         parent=styles['BodyText'],
#         fontSize=10,
#         leading=14,
#         textColor=colors.black,
#         spaceAfter=6,
#         alignment=TA_JUSTIFY
#     )
    
#     # Cover Page
#     elements.append(Spacer(1, 2*inch))
#     elements.append(Paragraph(f"Financial Analysis Report", title_style))
#     elements.append(Spacer(1, 0.5*inch))
#     elements.append(Paragraph(f"Prepared for: {company_name}", styles['Heading2']))
#     elements.append(Spacer(1, 0.25*inch))
#     elements.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", styles['BodyText']))
#     elements.append(PageBreak())
    
#     # Table of Contents
#     elements.append(Paragraph("Table of Contents", section_style))
#     toc = [
#         ("1. Executive Summary", "2"),
#         ("2. Financial Overview", "2"),
#         ("3. Performance Analysis", "3"),
#         ("4. Visualizations", "4"),
#         ("5. Forecast & Recommendations", str(len(image_paths)+4))
#     ]
    
#     toc_table = Table(toc, colWidths=[4*inch, 1*inch])
#     toc_table.setStyle(TableStyle([
#         ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
#         ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.grey),
#         ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
#     ]))
#     elements.append(toc_table)
#     elements.append(PageBreak())
    
#     # Executive Summary
#     elements.append(Paragraph("1. Executive Summary", section_style))
#     summary_text = f"""
#     This report provides a comprehensive analysis of {company_name}'s financial performance, comparing the current fiscal year with the previous year. 
#     The analysis includes detailed examination of revenue trends, expense patterns, profitability metrics, and predictive insights for future performance.
#     """
#     elements.append(Paragraph(summary_text, body_style))
#     elements.append(Spacer(1, 0.25*inch))
    
#     # Key Findings
#     elements.append(Paragraph("Key Findings", styles['Heading3']))
#     findings = [
#         f"• Total Revenue: ${current_totals['Revenue']:,.2f} ({((current_totals['Revenue']-previous_totals['Revenue'])/previous_totals['Revenue'])*100:+.2f}% YoY)",
#         f"• Total Profit: ${current_totals['Profit']:,.2f} ({((current_totals['Profit']-previous_totals['Profit'])/previous_totals['Profit'])*100:+.2f}% YoY)",
#         f"• Predicted Next Year Revenue: ${next_year_revenue:,.2f} ({revenue_change:+.2f}% growth)",
#         f"• KPI Threshold: Profit Margin < {kpi_threshold*100:.0f}%"
#     ]
#     for finding in findings:
#         elements.append(Paragraph(finding, body_style))
#     elements.append(PageBreak())
    
#     # Financial Overview
#     elements.append(Paragraph("2. Financial Overview", section_style))
    
#     # Summary Table
#     summary_data = [
#         ['Metric', 'Current Year', 'Previous Year', 'Change (%)'],
#         ['Revenue', f"${current_totals['Revenue']:,.2f}", f"${previous_totals['Revenue']:,.2f}", 
#          f"{((current_totals['Revenue']-previous_totals['Revenue'])/previous_totals['Revenue'])*100:+.2f}%"],
#         ['Expenses', f"${current_totals['Expenses']:,.2f}", f"${previous_totals['Expenses']:,.2f}", 
#          f"{((current_totals['Expenses']-previous_totals['Expenses'])/previous_totals['Expenses'])*100:+.2f}%"],
#         ['Profit', f"${current_totals['Profit']:,.2f}", f"${previous_totals['Profit']:,.2f}", 
#          f"{((current_totals['Profit']-previous_totals['Profit'])/previous_totals['Profit'])*100:+.2f}%"]
#     ]
    
#     summary_table = Table(summary_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1*inch])
#     summary_table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, 0), 10),
#         ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
#         ('GRID', (0, 0), (-1, -1), 1, colors.grey)
#     ]))
#     elements.append(summary_table)
#     elements.append(Spacer(1, 0.25*inch))
    
#     # Performance Analysis
#     elements.append(Paragraph("3. Performance Analysis", section_style))
    
#     # Add visualizations to PDF
#     elements.append(Paragraph("4. Visualizations", section_style))
#     for i, image_path in enumerate(image_paths):
#         img = Image(image_path, width=6*inch, height=4*inch)
#         elements.append(img)
#         elements.append(Spacer(1, 0.25*inch))
#         if (i+1) % 2 == 0 and i < len(image_paths)-1:
#             elements.append(PageBreak())
    
#     # Forecast & Recommendations
#     elements.append(PageBreak())
#     elements.append(Paragraph("5. Forecast & Recommendations", section_style))
#     forecast_text = f"""
#     Based on the linear regression analysis of historical revenue data, the projected revenue for the next fiscal year is estimated at ${next_year_revenue:,.2f}, 
#     representing a {revenue_change:+.2f}% change from the current year. This forecast suggests {"growth" if revenue_change > 0 else "decline"} in the company's financial performance.
#     """
#     elements.append(Paragraph(forecast_text, body_style))
    
#     elements.append(Paragraph("Recommendations:", styles['Heading3']))
#     recommendations = [
#         "• Continue current growth strategies if revenue projection is positive",
#         "• Review expense management if profit margins are below threshold",
#         "• Investigate months with significant budget variances",
#         "• Consider diversifying revenue streams if growth is stagnant"
#     ]
#     for rec in recommendations:
#         elements.append(Paragraph(rec, body_style))
    
#     # Footer with page numbers
#     def add_page_numbers(canvas, doc):
#         page_num = canvas.getPageNumber()
#         text = f"Page {page_num}"
#         canvas.setFont("Helvetica", 9)
#         canvas.drawRightString(7.5*inch, 0.5*inch, text)
    
#     doc.build(elements, onFirstPage=add_page_numbers, onLaterPages=add_page_numbers)
    
#     buffer.seek(0)
#     return buffer

# # Main application UI
# col1, col2 = st.columns(2)

# with col1:
#     company_name = st.text_input("Enter Company Name", "Example Inc.", key="company_name")
#     kpi_threshold = st.slider(
#         "Set Profit Margin Threshold for KPI Alerts (%)", 
#         0, 100, 30,
#         help="Months with profit margins below this threshold will trigger alerts"
#     ) / 100

# with col2:
#     current_year_file = st.file_uploader(
#         "Upload Current Year Financial Data (CSV)", 
#         type="csv",
#         help="CSV should contain columns: Month, Revenue, Expenses, Profit"
#     )
#     previous_year_file = st.file_uploader(
#         "Upload Previous Year Financial Data (CSV)", 
#         type="csv",
#         help="CSV should contain same structure as current year data"
#     )

# if current_year_file and previous_year_file:
#     current_df = load_data(current_year_file, "Current Year")
#     previous_df = load_data(previous_year_file, "Previous Year")
    
#     if current_df is not None and previous_df is not None:
#         # Generate report content
#         with st.spinner('Generating financial analysis...'):
#             report_content, current_totals, previous_totals, next_year_revenue, revenue_change, current_df = generate_report_text(
#                 current_df, previous_df, company_name, kpi_threshold
#             )
            
#             # Generate visualizations
#             image_paths, temp_dir = generate_visualizations(current_df, previous_df, company_name, kpi_threshold)
            
#             # Generate PDF report
#             pdf_buffer = generate_pdf_report(
#                 company_name, report_content, current_totals, previous_totals, 
#                 next_year_revenue, revenue_change, current_df, image_paths, kpi_threshold
#             )
        
#         # Display results
#         st.success('Analysis complete!')
        
#         # Report section
#         with st.expander("📄 View Detailed Report Analysis", expanded=True):
#             st.subheader(f"{company_name} - Financial Performance Analysis")
#             st.text_area("Full Report Details", report_content, height=400)
        
#         # Visualizations section
#         st.markdown("---")
#         st.subheader("📊 Financial Visualizations")
        
#         # Display charts in tabs
#         tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#             "Revenue Trends", "Profit Comparison", "Expense Breakdown", 
#             "Revenue vs Expenses", "Profit Distribution", "Monthly Details"
#         ])
        
#         with tab1:
#             st.image(image_paths[0], use_container_width=True, caption="Monthly Revenue Trends Comparison")
#         with tab2:
#             st.image(image_paths[1], use_container_width=True, caption="Monthly Profit Comparison")
#         with tab3:
#             st.image(image_paths[2], use_container_width=True, caption="Current Year Expense Breakdown")
#         with tab4:
#             st.image(image_paths[3], use_container_width=True, caption="Revenue vs Expenses (Current Year)")
#         with tab5:
#             st.image(image_paths[4], use_container_width=True, caption="Profit Distribution Comparison")
#         with tab6:
#             st.image(image_paths[5], use_container_width=True, caption="Monthly Financial Details")
        
#         # Clean up image files after displaying them
#         for image_path in image_paths:
#             if os.path.exists(image_path):
#                 os.remove(image_path)
        
#         # Download section
#         st.markdown("---")
#         st.subheader("📥 Download Full Report")
        
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             st.download_button(
#                 label="Download PDF Report",
#                 data=pdf_buffer,
#                 file_name=f"{company_name.replace(' ', '_')}_Financial_Report.pdf",
#                 mime="application/pdf",
#                 help="Generate a professional PDF report with all analysis and visualizations"
#             )
# else:
#     st.warning("⚠️ Please upload both current and previous year financial data to generate the report")