# Required imports
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Alpaca trading imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus, AssetExchange
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

trading_client = TradingClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET, paper = True)
#@tool
def scan_momentum_stocks() -> str:
    """
    Scans for momentum stock opportunities based on Warrior Trading criteria.
    Finds stocks with price $2-$20, up 30%+, 5x+ relative volume, and float ≤ 100M.
    
    Returns:
        Summary of momentum stocks found or error message
    """
    try:
        # Get all active, tradable assets
        request_params = GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass.US_EQUITY
        )
        assets = trading_client.get_all_assets(request_params)
        
        # Filter for NASDAQ and NYSE exchanges
        tradable_assets = [
            asset.symbol for asset in assets
            if asset.exchange in [AssetExchange.NASDAQ, AssetExchange.NYSE] and asset.tradable
        ]
        
        momentum_stocks = []
        print(f"Scanning {len(tradable_assets)} tradable assets for momentum opportunities...")
        
        # Process in batches of 25 to respect rate limits
        for i in range(0, len(tradable_assets), 25):
            batch = tradable_assets[i:i+25]
            print(f"Processing batch {i//25 + 1}/{(len(tradable_assets) + 24)//25}")
            
            # For each symbol, check momentum criteria
            for symbol in batch:
                try:
                    # Get current price
                    price_info = get_current_quote(symbol)
                    if "current price is $" not in price_info:
                        continue
                    
                    # Extract price from the response
                    price_str = price_info.split("current price is $")[1].split(" ")[0]
                    curr_price = float(price_str)
                    
                    # Skip if price not in our range
                    if not (2 <= curr_price <= 20):
                        continue
                    
                    # Get price change percentage
                    change_info = get_price_change(symbol)
                    if "up" not in change_info:
                        continue  # Skip stocks that aren't up
                    
                    # Extract percent change from the response
                    change_str = change_info.split("up ")[1].split("%")[0]
                    pct_change = float(change_str) / 100
                    
                    # Skip if percent change doesn't meet criteria
                    if pct_change < 0.30:  # 30% minimum gain
                        continue
                    
                    # Get relative volume
                    rvol_info = get_rvol(symbol)
                    if "RVOL for" not in rvol_info:
                        continue
                    
                    # Extract RVOL from the response
                    rvol_str = rvol_info.split("RVOL for")[1].split(":")[1].split(" ")[1]
                    rvol = float(rvol_str)
                    
                    # Skip if RVOL doesn't meet criteria
                    if rvol < 5:  # 5x minimum relative volume
                        continue
                    
                    # Get float information
                    float_info = get_float_info(symbol)
                    float_value = None
                    
                    if float_info and isinstance(float_info, str) and "{" in float_info:
                        # Try to parse the float value from the response
                        try:
                            import ast
                            float_data = ast.literal_eval(float_info)
                            float_value = float_data.get("Float")
                        except:
                            pass
                    
                    # Skip if float is too large
                    if float_value and float_value > 100_000_000:  # 100M maximum float
                        continue
                    
                    # All criteria met, add to momentum stocks
                    momentum_stocks.append({
                        'symbol': symbol,
                        'price': curr_price,
                        'pct_change': pct_change,
                        'rvol': rvol,
                        'float': float_value
                    })
                
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                    continue
            
            # Avoid hitting rate limits
            time.sleep(1)
        
        # Sort by percent change descending
        momentum_stocks.sort(key=lambda x: x['pct_change'], reverse=True)
        
        if not momentum_stocks:
            return "No momentum stocks found matching criteria at this time."
        
        # Format results nicely for chat display
        results = "🔥 MOMENTUM STOCKS FOUND 🔥\n\n"
        results += "Criteria: $2-$20 price, ≥30% gain, ≥5x volume, ≤100M float\n\n"
        
        for i, stock in enumerate(momentum_stocks[:10], 1):  # Top 10 only
            results += f"{i}. {stock['symbol']} - ${stock['price']:.2f}\n"
            results += f"   • Change: +{stock['pct_change']*100:.1f}%\n"
            results += f"   • RVOL: {stock['rvol']:.1f}x\n"
            
            if stock['float']:
                float_str = f"{stock['float']/1_000_000:.1f}M shares"
            else:
                float_str = "N/A"
            
            results += f"   • Float: {float_str}\n\n"
            
        results += f"Total matching stocks: {len(momentum_stocks)}"
        return results
    
    except Exception as e:
        return f"Error scanning for momentum stocks: {str(e)}"

#@tool
def get_top_momentum_stocks(top_n: int = 5) -> str:
    """
    Returns the top N momentum stocks that meet Warrior Trading criteria:
    - Price $2-$20
    - Up 30%+ today
    - 5x+ relative volume
    - Float ≤ 100M shares
    
    Args:
        top_n: Number of top stocks to return (default: 5)
    
    Returns:
        Summary of top momentum stocks or error message
    """
    try:
        full_results = scan_momentum_stocks()
        
        if "No momentum stocks found" in full_results:
            return full_results
            
        # Parse the full results to extract just the top N
        all_lines = full_results.split('\n')
        header_lines = all_lines[:3]  # Keep the header
        
        # Find where "Total matching stocks" line is
        footer_index = next((i for i, line in enumerate(all_lines) if "Total matching stocks" in line), -1)
        footer_line = [all_lines[footer_index]] if footer_index >= 0 else []
        
        # Extract just the top N stocks (each stock takes 4 lines)
        stock_blocks = []
        current_block = []
        stock_count = 0
        
        for line in all_lines[3:footer_index]:
            current_block.append(line)
            
            if line.strip() == "":  # End of a stock block
                if current_block and stock_count < top_n:
                    stock_blocks.extend(current_block)
                    stock_count += 1
                current_block = []
        
        # Combine all parts
        result_lines = header_lines + stock_blocks + footer_line
        return "\n".join(result_lines)
    
    except Exception as e:
        return f"Error retrieving top momentum stocks: {str(e)}"