from clients import llm, get_llm_with_tools, get_tool_node

# Review Agent's historical data retrieval tool
@tool
def get_historical_data_for_review(ticker: str, trade_date: str) -> str:
    """
    Retrieves historical data with technical indicators for the Review Agent to analyze.
    Gets enough historical data for indicators to be meaningful (50 days) but focuses
    on the specific trade date for analysis.
    
    Args:
        ticker: Stock symbol to analyze
        trade_date: The specific date to analyze (YYYY-MM-DD format)
        
    Returns:
        Summary of historical data with focus on the trade date
    """
    try:
        # Calculate start date (50 days before trade date for indicator calculation)
        trade_date_obj = datetime.strptime(trade_date, '%Y-%m-%d')
        start_date_obj = trade_date_obj - timedelta(days=50)
        start_date = start_date_obj.strftime('%Y-%m-%d')
        
        # Get historical data
        print(f"Fetching historical data for {ticker} from {start_date} to {trade_date}")
        data = get_alpaca_data(
            ticker=ticker,
            start_date=start_date,
            end_date=trade_date,
            timescale="Minute"
        )
        
        if data is None or data.empty:
            return f"Error: Could not retrieve historical data for {ticker}"
            
        # Add technical indicators
        data = add_indicators(data=data, indicator_set='alternate')
        
        # Get the specific trade day data
        trade_day_data = data[data.index.date == trade_date_obj.date()]
        
        if trade_day_data.empty:
            return f"Error: No data found for {ticker} on {trade_date}"
            
        # Calculate key metrics for the trade day
        day_open = trade_day_data.iloc[0]['Open']
        day_close = trade_day_data.iloc[-1]['Close']
        day_high = trade_day_data['High'].max()
        day_low = trade_day_data['Low'].min()
        day_volume = trade_day_data['Volume'].sum()
        
        # Calculate percent change
        day_change_pct = ((day_close - day_open) / day_open) * 100
        
        # Get some indicator values at key times
        first_hour_data = trade_day_data.iloc[:60] if len(trade_day_data) > 60 else trade_day_data
        
        # Format the response
        response = f"""
Historical Data Analysis for {ticker} on {trade_date}:

Daily Summary:
- Open: ${day_open:.2f}
- High: ${day_high:.2f}
- Low: ${day_low:.2f}
- Close: ${day_close:.2f}
- Volume: {day_volume:,}
- Day Change: {day_change_pct:.2f}%

First Hour Price Action:
{first_hour_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line']].head(10)}

Key Indicator Ranges:
- RSI Range: {trade_day_data['RSI'].min():.2f} - {trade_day_data['RSI'].max():.2f}
- MACD Range: {trade_day_data['MACD'].min():.4f} - {trade_day_data['MACD'].max():.4f}

Full Trade Day Data Points: {len(trade_day_data)}
Time Range: {trade_day_data.index[0]} to {trade_day_data.index[-1]}
"""
        
        # Store the data in a way the Review Agent can access for detailed analysis
        # This is a bit of a hack but allows the Review Agent to request specific time ranges
        response += f"\n[Data stored for detailed analysis - {len(trade_day_data)} minute bars available]"
        
        return response
        
    except Exception as e:
        return f"Error retrieving historical data: {str(e)}"


# Review Agent System Prompt
REVIEW_AGENT_SYSTEM_PROMPT = (
    "system",
    """
You are a Trade Review Agent specializing in analyzing historical trading opportunities using the Micro Pullback strategy.

Your primary responsibilities:
1. Analyze YouTube trading recap videos to extract tickers and trade dates
2. Review historical market data for identified stocks
3. Determine optimal entry and exit points using the Micro Pullback strategy
4. Generate simulation parameters for testing

MICRO PULLBACK STRATEGY CRITERIA:

Entry Requirements:
- Stock must be up at least 10% from previous day's close
- Relative volume (RVOL) should be high (ideally 5x+ normal)
- Price range typically $1-$20 (though you'll analyze any ticker from videos)
- Look for momentum followed by a small pullback
- Volume should increase as price resumes upward movement
- MACD should be bullish, RSI not extremely overbought (under 80)

Entry Signal:
- After initial momentum move (1%+ in recent candles)
- Small red candle or lower wick appears (the pullback)
- Price holds above key support levels
- Volume starts increasing again
- Price breaks above the high of the pullback candle

Exit Criteria:
- Price drops below the low of the pullback (stop loss)
- MACD crosses below signal line with momentum fading
- Formation of rejection candle (long upper wick)
- Reaches next psychological level (half/whole dollar)
- Achieved 10%+ gain from entry

When analyzing historical data:
1. Focus on the first 2-3 hours of trading (most momentum plays happen early)
2. Identify all potential micro pullback setups
3. Select the highest probability setup based on the criteria
4. Note the exact entry time and price
5. Determine the exit based on the first exit signal that appears

Output simulation parameters in this format:
{
    "ticker": "SYMBOL",
    "date": "YYYY-MM-DD",
    "expected_entry": {
        "time": "HH:MM:SS",
        "price": 0.00,
        "reasoning": "Brief explanation"
    },
    "expected_exit": {
        "time": "HH:MM:SS", 
        "price": 0.00,
        "reasoning": "Brief explanation"
    },
    "trade_quality": "High/Medium/Low",
    "notes": "Any important observations"
}

Remember: You're reconstructing what the optimal trade SHOULD have been based on the Micro Pullback strategy, not trying to match exactly what happened in the video.
"""
)

### Review Agent Setup
from langchain_core.messages import SystemMessage

# Create the Review Agent with tools
review_tools = [
    analyze_trade_video_from_url,  # The video analysis tool
    get_historical_data_for_review  # The historical data tool
]
class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    finished: bool
    query: list[str]

def maybe_route_to_reviewtools(state: ReviewState) -> Literal["review_tools", "human"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "review_tools"
    return "human"

def human_node(state: ReviewState) -> ReviewState:
    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "content"):
        print("🧠 Gemini:", last.content)
    
    user_input = input("You: ")
    if user_input.lower() in {"q", "quit", "exit"}:
        return state | {"finished": True}

    return state | {"messages": [HumanMessage(content=user_input)]}

def gemini_review_node(state: ReviewState) -> ReviewState:
    sysmsg = SystemMessage(content=REVIEW_AGENT_SYSTEM_PROMPT[1])
    history = [sysmsg] + state["messages"]
    
    if state["messages"]:
        new_output = review_llm_with_tools.invoke(history)
    else:
        new_output = AIMessage(content="Ready to evaluate stocks. Ask me which ticker to analyze.")

    return state | {"messages": [new_output]}

review_tool_node = get_tool_node(review_tools)

# Initialize the Review Agent
review_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")#ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.environ["GOOGLE_API_KEY"])#
review_llm_with_tools = get_llm_with_tools(review_tools)

review_graph = StateGraph(ReviewState)
review_graph.add_node("review_tools", review_tool_node)

review_graph.add_node("gemini", gemini_review_node)
review_graph.add_node("human", human_node)


review_graph.add_conditional_edges("gemini", maybe_route_to_reviewtools)
review_graph.add_conditional_edges("human", lambda state: END if state.get("finished") else "gemini")
review_graph.add_edge("review_tools", "gemini")
review_graph.set_entry_point("human")

compiled_review_graph = review_graph.compile()


config = {"recursion_limit": 100}
print("🧠 Gemini Trading Agent Ready. Type 'q' to quit.")

compiled_review_graph.invoke({"messages": [], "query": [], "finished": False}, config)