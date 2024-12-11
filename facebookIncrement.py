import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import os
import json
from datetime import datetime

# Set the style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

def create_market_directory(market_name):
    """
    Create a directory for the market if it doesn't exist
    """
    dir_name = f"market_analysis_{market_name.lower().replace(' ', '_')}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def load_data():
    """
    Load data from the market-specific Excel file
    """
    try:
        # Use the market-specific file
        file_name = 'data_template_wb_2years_bymarket_fixed.xlsx'
        
        if not os.path.exists(file_name):
            print(f"Error: {file_name} not found in the directory")
            return None

        print(f"Using file: {file_name}")

        # Load the data
        data = pd.read_excel(file_name)

        # Remove rows where essential data is missing
        required_cols = ['Date', 'fb_spend', 'shopify_revenue', 'market']
        if not all(col in data.columns for col in required_cols):
            print(f"Error: Required columns {required_cols} not found in the spreadsheet")
            return None

        # Convert Date column to datetime if it's not already
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Replace any negative values with 0
        data['fb_spend'] = data['fb_spend'].clip(lower=0)
        data['shopify_revenue'] = data['shopify_revenue'].clip(lower=0)
        
        # Remove rows where both spend and revenue are 0
        data = data[(data['fb_spend'] > 0) | (data['shopify_revenue'] > 0)]
        
        # Sort by date
        data = data.sort_values('Date')
        
        # Add time-based features
        data['day_of_week'] = data['Date'].dt.day_name()
        data['month'] = data['Date'].dt.month
        data['year'] = data['Date'].dt.year
        data['quarter'] = data['Date'].dt.quarter
        data['is_weekend'] = data['Date'].dt.weekday.isin([5, 6])
        
        # Calculate rolling metrics (7-day and 30-day)
        for market in data['market'].unique():
            mask = data['market'] == market
            data.loc[mask, 'rolling_7d_spend'] = data.loc[mask, 'fb_spend'].rolling(7, min_periods=1).mean()
            data.loc[mask, 'rolling_7d_revenue'] = data.loc[mask, 'shopify_revenue'].rolling(7, min_periods=1).mean()
            data.loc[mask, 'rolling_30d_spend'] = data.loc[mask, 'fb_spend'].rolling(30, min_periods=1).mean()
            data.loc[mask, 'rolling_30d_revenue'] = data.loc[mask, 'shopify_revenue'].rolling(30, min_periods=1).mean()

        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_incrementality(data):
    """
    Analyze the incrementality of Facebook spend on revenue
    """
    if data is None or len(data) == 0:
        print("No data available for analysis")
        return None

    # Calculate daily metrics with proper handling of zero spend
    data['revenue_per_spend'] = np.where(data['fb_spend'] > 0, 
                                       data['shopify_revenue'] / data['fb_spend'], 
                                       0)  # Set ROAS to 0 when spend is 0
    
    data['ctr'] = data['fb_clicks'] / data['fb_impressions'] if 'fb_clicks' in data.columns else None
    data['conversion_rate'] = data['fb_conversions'] / data['fb_clicks'] if 'fb_conversions' in data.columns else None

    # Basic correlation analysis
    correlation = data['fb_spend'].corr(data['shopify_revenue'])

    # Simple linear regression
    X = data['fb_spend'].values.reshape(-1, 1)
    y = data['shopify_revenue'].values

    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    coefficient = model.coef_[0]

    # Calculate overall ROAS safely
    total_spend = data['fb_spend'].sum()
    total_revenue = data['shopify_revenue'].sum()
    overall_roas = total_revenue / total_spend if total_spend > 0 else 0

    # Calculate metrics by time period
    yearly_metrics = data.groupby('year').agg({
        'fb_spend': 'sum',
        'shopify_revenue': 'sum'
    })
    yearly_metrics['revenue_per_spend'] = np.where(yearly_metrics['fb_spend'] > 0,
                                                 yearly_metrics['shopify_revenue'] / yearly_metrics['fb_spend'],
                                                 0)

    quarterly_metrics = data.groupby(['year', 'quarter']).agg({
        'fb_spend': 'sum',
        'shopify_revenue': 'sum'
    })
    quarterly_metrics['revenue_per_spend'] = np.where(quarterly_metrics['fb_spend'] > 0,
                                                    quarterly_metrics['shopify_revenue'] / quarterly_metrics['fb_spend'],
                                                    0)

    monthly_metrics = data.groupby(['year', 'month']).agg({
        'fb_spend': 'sum',
        'shopify_revenue': 'sum'
    })
    monthly_metrics['revenue_per_spend'] = np.where(monthly_metrics['fb_spend'] > 0,
                                                  monthly_metrics['shopify_revenue'] / monthly_metrics['fb_spend'],
                                                  0)

    dow_metrics = data.groupby('day_of_week').agg({
        'fb_spend': 'mean',
        'shopify_revenue': 'mean'
    })
    dow_metrics['revenue_per_spend'] = np.where(dow_metrics['fb_spend'] > 0,
                                              dow_metrics['shopify_revenue'] / dow_metrics['fb_spend'],
                                              0)

    return {
        'correlation': correlation,
        'r_squared': r_squared,
        'coefficient': coefficient,
        'overall_roas': overall_roas,
        'total_revenue': total_revenue,
        'total_spend': total_spend,
        'yearly_metrics': yearly_metrics.to_dict(),
        'quarterly_metrics': quarterly_metrics.to_dict(),
        'monthly_metrics': monthly_metrics.to_dict(),
        'dow_metrics': dow_metrics.to_dict(),
        'start_date': data['Date'].min(),
        'end_date': data['Date'].max()
    }

def plot_market_analysis(data, market_name, market_dir):
    """
    Create individual visualizations for the analysis for a specific market
    """
    try:
        # Check if there's any spend data
        has_spend = (data['fb_spend'] > 0).any()
        
        # Spend vs Revenue Scatter Plot
        fig, ax1 = plt.subplots(figsize=(15, 8))
        sns.scatterplot(data=data, x='fb_spend', y='shopify_revenue', ax=ax1)
        ax1.set_title(f'Facebook Spend vs Revenue - {market_name}')
        ax1.set_xlabel('Facebook Spend ($)')
        ax1.set_ylabel('Revenue ($)')
        
        # Add trend line only if there's spend data and more than one point
        if has_spend and len(data) > 1:
            mask = data['fb_spend'] > 0
            if mask.sum() > 1:  # Need at least 2 points for a trend line
                z = np.polyfit(data.loc[mask, 'fb_spend'], data.loc[mask, 'shopify_revenue'], 1)
                p = np.poly1d(z)
                plot_x = data.loc[mask, 'fb_spend']
                ax1.plot(plot_x, p(plot_x), "r--", alpha=0.8, label='Trend Line')
                ax1.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(market_dir, 'spend_vs_revenue.png'))
        plt.close()

        # Time series plot with 7-day rolling metrics
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['rolling_7d_spend'], label='7-day Rolling Spend')
        plt.plot(data['Date'], data['rolling_7d_revenue'], label='7-day Rolling Revenue')
        plt.title(f'7-day Rolling Metrics - {market_name}')
        plt.xlabel('Date')
        plt.ylabel('Amount ($)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(market_dir, 'rolling_metrics.png'))
        plt.close()

        # Monthly performance
        monthly_data = data.groupby(['year', 'month']).agg({
            'fb_spend': 'sum',
            'shopify_revenue': 'sum'
        }).reset_index()
        monthly_data['month_year'] = monthly_data.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02}", axis=1)
        monthly_data['roas'] = np.where(monthly_data['fb_spend'] > 0,
                                      monthly_data['shopify_revenue'] / monthly_data['fb_spend'],
                                      0)

        plt.figure(figsize=(15, 6))
        ax = plt.gca()
        monthly_data.plot(kind='bar', x='month_year', y=['fb_spend', 'shopify_revenue'], ax=ax)
        plt.title(f'Monthly Performance - {market_name}')
        plt.xlabel('Month')
        plt.ylabel('Amount ($)')
        plt.xticks(rotation=45)
        plt.legend(['Spend', 'Revenue'])
        plt.tight_layout()
        plt.savefig(os.path.join(market_dir, 'monthly_performance.png'))
        plt.close()

        # Day of week analysis - only create if there's spend data
        if has_spend:
            dow_data = data.groupby('day_of_week').agg({
                'fb_spend': 'mean',
                'shopify_revenue': 'mean'
            }).reset_index()
            dow_data['roas'] = np.where(dow_data['fb_spend'] > 0,
                                      dow_data['shopify_revenue'] / dow_data['fb_spend'],
                                      0)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=dow_data, x='day_of_week', y='roas')
            plt.title(f'Average ROAS by Day of Week - {market_name}')
            plt.xlabel('Day of Week')
            plt.ylabel('ROAS')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(market_dir, 'dow_roas.png'))
            plt.close()
        else:
            # Create a text file explaining why ROAS analysis is not available
            with open(os.path.join(market_dir, 'no_roas_analysis.txt'), 'w') as f:
                f.write(f"ROAS analysis is not available for {market_name} because there was no Facebook spend during the analysis period.")

    except Exception as e:
        print(f"Error creating plots for {market_name}: {e}")
        # Create error log file
        with open(os.path.join(market_dir, 'error_log.txt'), 'w') as f:
            f.write(f"Error occurred while creating plots: {str(e)}")

def create_detailed_market_analysis(metrics, market):
    """
    Create a detailed analysis of the market's performance including statistical interpretation
    """
    analysis = f"\n{'='*50}\n"
    analysis += f"Detailed Analysis for {market}\n"
    analysis += f"{'='*50}\n\n"
    
    # Basic Information
    analysis += f"Analysis Period: {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')}\n\n"
    
    # Financial Performance
    analysis += "1. Financial Performance\n"
    analysis += "------------------------\n"
    analysis += f"Total Revenue: ${metrics['total_revenue']:,.2f}\n"
    analysis += f"Total Facebook Spend: ${metrics['total_spend']:,.2f}\n"
    analysis += f"Overall ROAS: ${metrics['overall_roas']:.2f}\n\n"
    
    # Statistical Analysis
    analysis += "2. Statistical Analysis\n"
    analysis += "------------------------\n"
    
    # Correlation Interpretation
    analysis += "Correlation Analysis:\n"
    correlation = metrics['correlation']
    if pd.isna(correlation):
        analysis += "- No correlation could be calculated (insufficient variation in spend)\n"
    else:
        analysis += f"- Correlation coefficient: {correlation:.3f}\n"
        if abs(correlation) < 0.3:
            analysis += "- Interpretation: Weak relationship between spend and revenue\n"
        elif abs(correlation) < 0.7:
            analysis += "- Interpretation: Moderate relationship between spend and revenue\n"
        else:
            analysis += "- Interpretation: Strong relationship between spend and revenue\n"
    
    # R-squared Interpretation
    analysis += "\nModel Fit:\n"
    r_squared = metrics['r_squared']
    analysis += f"- R-squared value: {r_squared:.3f}\n"
    if r_squared < 0.3:
        analysis += "- Interpretation: The model explains a small portion of revenue variance\n"
    elif r_squared < 0.7:
        analysis += "- Interpretation: The model explains a moderate portion of revenue variance\n"
    else:
        analysis += "- Interpretation: The model explains a large portion of revenue variance\n"
    
    # Revenue per Dollar Analysis
    analysis += "\nRevenue Impact:\n"
    coefficient = metrics['coefficient']
    analysis += f"- For every $1 spent on Facebook, revenue changes by ${coefficient:.2f}\n"
    if coefficient <= 0:
        analysis += "- Warning: Negative or zero revenue impact suggests ineffective spend\n"
    elif coefficient < 1:
        analysis += "- Warning: Revenue increase is less than spend amount\n"
    else:
        analysis += "- Positive ROI: Each dollar spent generates more than $1 in revenue\n"
    
    # Recommendations
    analysis += "\n3. Recommendations\n"
    analysis += "------------------------\n"
    if metrics['overall_roas'] == 0:
        analysis += "- Market currently shows no Facebook spend - consider testing with small budget\n"
    elif metrics['overall_roas'] < 1:
        analysis += "- Urgent review needed: Current strategy is not profitable\n"
    elif correlation < 0:
        analysis += "- Review strategy: Negative correlation suggests potential issues\n"
    elif r_squared < 0.1:
        analysis += "- Consider testing different approaches: Current spend has unpredictable results\n"
    else:
        if metrics['overall_roas'] > 2:
            analysis += "- Consider increasing budget given strong ROAS\n"
        analysis += "- Continue monitoring and optimizing campaigns\n"
    
    analysis += "\n"
    return analysis

def main():
    """
    Main function to run the analysis
    """
    # Load data
    print("Loading data...")
    data = load_data()
    
    if data is None:
        print("Error: Could not load data")
        return

    # Create summary for all markets
    print(f"\nAnalyzing {len(data['market'].unique())} markets...\n")
    overall_summary = ""
    detailed_analysis = "Facebook Incrementality Analysis - Detailed Market Analysis\n"
    detailed_analysis += "================================================\n"
    
    for market in data['market'].unique():
        print(f"\nAnalyzing market: {market}")
        
        # Create market directory
        market_dir = create_market_directory(market)
        
        # Filter data for this market
        market_data = data[data['market'] == market].copy()
        
        # Generate metrics
        metrics = analyze_incrementality(market_data)
        
        if metrics is not None:
            # Print analysis results
            print(f"\nIncrementality Analysis Results for {market}:")
            print(f"Analysis Period: {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')}")
            print(f"Correlation between spend and revenue: {metrics['correlation']:.3f}")
            print(f"R-squared value: {metrics['r_squared']:.3f}")
            print(f"For every $1 spent on Facebook, revenue increases by ${metrics['coefficient']:.2f}")
            print(f"Overall ROAS: ${metrics['overall_roas']:.2f}")
            
            # Generate plots
            plot_market_analysis(market_data, market, market_dir)
            
            # Add to overall summary
            overall_summary += f"\nMarket: {market}\n"
            overall_summary += f"  Total Revenue: ${metrics['total_revenue']:,.2f}\n"
            overall_summary += f"  Total Spend: ${metrics['total_spend']:,.2f}\n"
            overall_summary += f"  Overall ROAS: ${metrics['overall_roas']:.2f}\n"
            overall_summary += f"  Revenue per $1 spent: ${metrics['coefficient']:.2f}\n"
            overall_summary += f"  Correlation: {metrics['correlation']:.3f}\n\n"
            
            # Add detailed analysis
            detailed_analysis += create_detailed_market_analysis(metrics, market)
    
    # Save overall summary
    with open('all_markets_summary.txt', 'w') as f:
        f.write(overall_summary)
    
    # Save detailed analysis
    with open('detailed_market_analysis.txt', 'w') as f:
        f.write(detailed_analysis)
    
    print("\nAnalysis complete! Check the market_analysis_* directories for detailed results.")

if __name__ == "__main__":
    main()