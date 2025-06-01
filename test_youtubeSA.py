youtube_url = "https://youtu.be/p2uUTz_StZQ?si=lvfb1fBPNo5cCoFJ"

print("Analyzing video... (this may take a minute)")
analysis_result = analyze_trade_video_from_url(youtube_url)

result_data = json.loads(analysis_result)
#print(result_data)
# Display the results nicely formatted
if result_data.get("success"):
    print("\n" + "="*80)
    print(f"✅ ANALYSIS COMPLETE: {result_data.get('tickers')} trade on {result_data.get('trade_date')}")
    print("="*80)
    
    # Basic trade info
    print(f"\n📊 TICKER: {result_data['tickers']}")
    print(f"📅 TRADE DATE: {result_data['trade_date']}")
    
    # Confidence scores
    analysis = result_data['trade_analysis']
        
    # Trade details
    full_summary = analysis['summary'].get('full_summary')