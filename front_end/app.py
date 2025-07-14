from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import os

app = Flask(__name__)

# Load the datasets
comparison_df = pd.read_csv("data/november_predictions.csv")
error_df = pd.read_csv("data/model_error_summary.csv")
clean_df = pd.read_csv("data/cleaned_daily_power.csv")

comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])
clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'])
clean_df['month_name'] = clean_df['timestamp'].dt.month_name()
clean_df['day_of_week'] = clean_df['timestamp'].dt.day_name()

model_column_map = {
    "RandomForest": "RandomForest_Predicted",
    "Prophet": "Prophet_Predicted",
    "XGBoost": "XGBoost_Predicted",
    "LightGBM": "LightGBM_Predicted"
}

model_error_name_map = {
    "RandomForest": "Random Forest",
    "Prophet": "Prophet",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM"
}

@app.route('/')
def index():
    selected_model = request.args.get('model', 'RandomForest')
    selected_month = request.args.get('month', 'All')

    selected_column = model_column_map.get(selected_model)
    if selected_column not in comparison_df.columns:
        return f"Invalid model selection: {selected_model}", 400

    filtered_df = comparison_df.copy()
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['Date'].dt.month_name() == selected_month]

    # Model Prediction Chart
    trace1 = go.Scatter(x=filtered_df['Date'], y=filtered_df['Actual'], mode='lines', name='Actual', line=dict(color='green'))
    trace2 = go.Scatter(x=filtered_df['Date'], y=filtered_df[selected_column], mode='lines', name=selected_model, line=dict(color='blue'))
    model_fig = go.Figure([trace1, trace2])
    model_fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=30))
    model_chart = pio.to_html(model_fig, full_html=False)

    # RMSE & MAPE values
    csv_model_name = model_error_name_map.get(selected_model, selected_model)
    m_row = error_df[error_df['Model'] == csv_model_name]
    mape = round(m_row['MAPE'].values[0], 2) if not m_row.empty else 'N/A'
    rmse = round(m_row['RMSE'].values[0], 2) if not m_row.empty else 'N/A'

    # Daily Solar Generation
    daily_df = clean_df.copy()
    if selected_month != 'All':
        daily_df = daily_df[daily_df['month_name'] == selected_month]
    daily_group = daily_df.groupby(daily_df['timestamp'].dt.date)['Power(MW)'].mean()
    daily_fig = go.Figure([go.Scatter(x=daily_group.index, y=daily_group.values, mode='lines+markers', line=dict(color='orange'))])
    daily_fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=30))
    daily_chart = pio.to_html(daily_fig, full_html=False)

    # Monthly Avg Power
    monthly_avg = clean_df.groupby(clean_df['timestamp'].dt.month_name())['Power(MW)'].mean().sort_index()
    monthly_fig = go.Figure([go.Bar(x=monthly_avg.index, y=monthly_avg.values, marker_color='skyblue')])
    monthly_fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=30))
    monthly_chart = pio.to_html(monthly_fig, full_html=False)

    # Power by Day of Week
    dow_avg = clean_df.groupby('day_of_week')['Power(MW)'].mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    dow_fig = go.Figure([go.Bar(x=dow_avg.index, y=dow_avg.values, marker_color='mediumpurple')])
    dow_fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=30))
    dayofweek_chart = pio.to_html(dow_fig, full_html=False)

    # RMSE vs MAPE Model Comparison
    err_fig = go.Figure()
    err_fig.add_trace(go.Bar(x=error_df['Model'], y=error_df['MAPE'], name='MAPE (%)', marker_color='lightgreen'))
    err_fig.add_trace(go.Bar(x=error_df['Model'], y=error_df['RMSE'], name='RMSE', marker_color='steelblue'))
    err_fig.update_layout(barmode='group', height=300, margin=dict(l=20, r=20, t=30, b=30))
    error_chart = pio.to_html(err_fig, full_html=False)

    return render_template('index.html', 
        selected_model=selected_model, 
        selected_month=selected_month,
        mape=mape,
        rmse=rmse,
        model_chart=model_chart,
        daily_chart=daily_chart,
        monthly_chart=monthly_chart,
        dayofweek_chart=dayofweek_chart,
        error_chart=error_chart)

if __name__ == '__main__':
    app.run(debug=True)
