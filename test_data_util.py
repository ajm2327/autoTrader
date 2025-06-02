import sys
import os
import traceback
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_util import get_alpaca_data, add_indicators, dataframe_info
    from plotting_utils import (
        plot_price_data, plot_with_moving_averages, plot_technical_indicators,
        plot_data_comparison, plot_volume_analysis, save_plot, show_data_summary
    )
    print(" Successfully imported all required modules")
except ImportError as e:
    print(f" Import Error: {e}")
    sys.exit(1)

def test_data_retrieval(ticker: str, days_back: int = 30, timescale: str= 'Day') -> pd.DataFrame:
    print(f"\n Testing Data Retrieval for {ticker}")
    print(f"Parameters: {days_back} days back, timescale: {timescale}")
    print("-" * 50)
    try:
        # calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        print(f"Date Range: {start_date} to {end_date}")

        data = get_alpaca_data(ticker = ticker, start_date=start_date, end_date=end_date, timescale=timescale)

        if data is None or data.empty:
            print(f"Failed to retrieve data for {ticker}")
            return None
        print(f"** SUCCESS Retrieved {len(data)} data points ")
        show_data_summary(data, ticker)

        if not validate_data_integrity(data, ticker):
            print("XXX Data integrity issues detected")
            return None
        
        print(" Data integrity validation passed")
        return data
    
    except Exception as e:
        print(f" XXX Error in data retrieval: {str(e)}")
        traceback.print_exc()
        return None
    

def test_indicator_calculation(data: pd.DataFrame, ticker: str, indicator_set: str = 'alternate') -> pd.DataFrame:
    print(f"\n TESTING INDICATOR CALCULATION for {ticker}")
    print(f"Using indicator set: {indicator_set}")
    print("-" * 50)

    try: 
        # make a copy to preserve original data
        data_copy = data.copy()
        # add indicators
        enhanced_data = add_indicators(data_copy, indicator_set= indicator_set)
        if enhanced_data is None:
            print(f" XXX Failed to add indicators for {ticker}")
            return None
        
        
        # Show whats added
        original_cols = set(data.columns)
        new_cols = set(enhanced_data.columns) - original_cols
        print(f" ** SUCCESS added {len(new_cols)} indicators")
        print(f"New Columns: {', '.join(sorted(new_cols))}")

        # validate indicators
        if not validate_indicators(enhanced_data, indicator_set):
            print('XXX Indicator validation failed')
            return None
        
        print("** Indicator validation passed")
        show_data_summary(enhanced_data, ticker)
        return enhanced_data
    except Exception as e:
        print(f"XXX Error in indicator calculation: {str(e)}")
        traceback.print_exc()
        return None
    
def validate_data_integrity(data: pd.DataFrame, ticker: str) -> bool:
    print(f"VALIDATING Data integrity for {ticker}...")
    issues = []

    required_cols=['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        issues.append(f"Missing columns: {', '.join(missing_cols)}")
    # check for nulls
    null_counts = data.isnull().sum()
    if null_counts.sum() > 0:
        issues.append(f"Found null values: {null_counts[null_counts > 0].to_dict()}")

    # check price relationships
    if 'High' in data.columns and 'Low' in data.columns:
        invalid_hl = (data['High'] < data['Low']).sum()
        if invalid_hl > 0:
            issues.append(f"Found {invalid_hl} rows where High < Low")

    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in data.columns:
            negative_prices = (data[col] <= 0).sum()
            if negative_prices > 0:
                issues.append(f"Found {negative_prices} negative or zero values in {col}")
    if 'Volume' in data.columns:
        negative_vol = (data['Volume'] <= 0).sum()
        if negative_vol > 0:
            issues.append(f"Found {negative_vol} negative volume vals")

    if not isinstance(data.index, pd.DatetimeIndex):
        issues.append("Index is not a DatetimeIndex")

    if issues: 
        print(" XXX Data integrity issues found:")
        for issue in issues:
            print(f" - {issue}")
        return False
    print(" ** Data integrity validation passed")
    return True

def validate_indicators(data: pd.DataFrame, indicator_set: str) -> bool:
    print(f" VALIDATING Indicators for set: {indicator_set}...")
    issues = []

    if indicator_set == 'alternate':
        # Check RSI range (0-100)
        if 'RSI' in data.columns:
            rsi_out_of_range = ((data['RSI'] < 0) | (data['RSI'] > 100)).sum()
            if rsi_out_of_range > 0:
                issues.append(f"RSI values out of range: {rsi_out_of_range} rows")
        
        if 'SMA_20' in data.columns and 'Close' in data.columns:
            # SMA Should be close to price
            price_diff = abs(data['SMA_20'] - data['Close']) / data['Close']
            extreme_diff = (price_diff > 0.5).sum()
            if extreme_diff > 0:
                issues.append(f"Extreme SMA_20 vs Close Differences: {extrem_diff}")
        
        if 'MACD' in data.columns and 'Signal_Line' in data.columns:
            macd_null = data['MACD'].isnull().sum()
            signal_null = data['Signal_Line'].isnull().sum()
            if macd_null > len(data) * 0.8:
                issues.append(f"MACD has too many null values: {macd_null}")
            if signal_null > len(data) * 0.8:
                issues.append(f"Signal Line has too many null values: {signal_null}")
    if issues:
        print(" XXX Indicator validation issues found:")
        for issue in issues:
            print(f" - {issue}")
        return False
    print(" ** Indicator validation passed")
    return True

def create_test_plots(original_data: pd.DataFrame, enhanced_data: pd.DataFrame, ticker: str, output_dir: str = 'test_plots'):
    print(f"\n Creating test plots for {ticker}")
    print("-" * 50)

    os.makedirs(output_dir, exist_ok = True)
    try:
        print("Creating basic price data plot")
        fig1 = plot_price_data(original_data, ticker, 'Raw Data')
        save_plot(fig1, f"{output_dir}/{ticker}_01_raw_price_data.png")
        plt.close(fig1)

        if 'SMA_20' in enhanced_data.columns:
            print("Creating moving averages plot...")
            fig2 = plot_with_moving_averages(enhanced_data, ticker)
            save_plot(fig1, f"{output_dir}/{ticker}_02_moving_averages.png")
            plt.close(fig2)

        print("creating technical indicators plot...")
        fig3 = plot_technical_indicators(enhanced_data, ticker)
        save_plot(fig3, f"{output_dir}/{ticker}_03_technical_indicators.png")
        plt.close(fig3)

        print("Creating data comparison plot...")
        fig4 = plot_data_comparison(original_data, enhanced_data, ticker)
        save_plot(fig4, f"{output_dir}/{ticker}_04_data_comparison.png")
        plt.close(fig4)

        print("Creating volume analysis plot...")
        fig5 = plot_volume_analysis(enhanced_data, ticker)
        save_plot(fig5, f"{output_dir}/{ticker}_05_volume_analysis.png")
        plt.close(fig5)

        print(f"All plots saved to {output_dir}")

    except Exception as e:
        print(f"XXX Error creating plots: {str(e)}")
        traceback.print_exc()
        
def run_comprehensive_test(ticker: str = 'AMD', days_back: int=30, timescale: str = "Day",
                           indicator_set: str = 'alternate', create_plots: bool = True):
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST FOR {ticker}")
    print(f"{'='*60}")

    results = {
        'ticker': ticker,
        'data_retrieval': False,
        'indicator_calculation': False,
        'plots_created': False,
        'errors': []
    }

    try:
        original_data = test_data_retrieval(ticker, days_back, timescale)
        if original_data is not None:
            results['data_retrieval'] = True
        else:
            results['errors'].append("Data retrieval failed")
            return results
        
        enhanced_data = test_indicator_calculation(original_data, ticker, indicator_set)
        if enhanced_data is not None:
            results['indicator_calculation'] = True
        else:
            results['errors'].append("Indicator calculation failed")
            return results
        
        if create_plots:
            create_test_plots(original_data, enhanced_data, ticker)
            results['plots_created'] = True
        
        print(f"\n COMPREHENSIVE TEST SUCCESSFUL for {ticker}")

    except Exception as e:
        error_msg = f"XXX Error during comprehensive test: {str(e)}"
        print(error_msg)
        results['errors'].append(error_msg)
        traceback.print_exc()
    return results



def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AMD'
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print("DATA RETRIEVAL TEST")
    print(f"Testing with ticker: {ticker}, days back: {days_back}")

    if ',' in ticker:
        tickers = [t.strip().upper() for t in ticker.split(',')]
        print(f"Testing multiple tickers: {tickers}")

        all_results = {}
        for test_ticker in tickers:
            print(f"\n{'='*60}")
            print(f"Testing {test_ticker}")
            print(f"{'='*60}")

            results = run_comprehensive_test(
                ticker = test_ticker, 
                days_back=days_back,
                timescale='Day',
                indicator_set='alternate',
                create_plots = True)
            all_results[test_ticker] = results

        print(f"\n{'='*60}")
        print("COMPREHENSIVE TEST RESULTS:")
        print(f"{'='*60}")

        for test_ticker, result in all_results.items():
            status = "** PASS" if all([ 
            result['data_retrieval'],
            result['indicator_calculation'],
            result['plots_created']]) else 'XXX FAIL'

            print(f"{test_ticker}: {status}")
            if result['errors']:
                for error in result['errors']:
                    print(f" - {error}")
    else:
        results = run_comprehensive_test(
            ticker=ticker.upper(), 
            days_back=days_back,
            timescale='Day',
            indicator_set='alternate',
            create_plots=True
        )

        print(f"\n{'='*60}")
        print("Final Test Results:")
        print(f"{'='*60}")
        print(f"Ticker: {results['ticker']}")
        print(f"Data Retrieval: {'Pass' if results['data_retrieval'] else ' XXX FAIL'}")
        print(f"Indicator Calculation: {'Pass' if results['indicator_calculation'] else ' XXX FAIL'}")
        print(f"Plots Created: {'Pass' if results['plots_created'] else ' XXX FAIL'}")

        if results['errors']:
            print(f"\nErrors encoutnered:")
            for error in results['errors']:
                print(f" - {error}")
        else:
            print(f"\n SUCCESS All test passed")

if __name__ == "__main__":
    main()


