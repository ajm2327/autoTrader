import json
from IPython.display import display, HTML
import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import re
import json
from langchain_core.tools import tool

from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain_core.tools import tool

#@tool
def get_youtube_transcript_from_url(youtube_url: str) -> str:
    """
    Retrieves the transcript of a YouTube video with timestamps from a YouTube URL.
    
    Args:
        youtube_url: Full YouTube URL (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)
        
    Returns:
        Formatted transcript with timestamps or error message
    """
    try:
        # Extract video ID from URL
        video_id = extract_youtube_video_id(youtube_url)
        if not video_id:
            return f"Error: Could not extract video ID from URL: {youtube_url}"
        
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format transcript with timestamps
        full_transcript = ""
        for entry in transcript_list:
            timestamp = entry['start']
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"[{minutes:02d}:{seconds:02d}] "
            
            full_transcript += time_str + entry['text'] + "\n"
            
        return full_transcript
    except Exception as e:
        return f"Error retrieving transcript: {str(e)}"


def create_vector_store(transcript_text):
    """
    Creates a vector store from the transcript text for semantic search.
    
    Args:
        transcript_text: Full transcript text with timestamps
        
    Returns:
        Dictionary with vector store and metadata
    """
    # Optimized text splitter for trading transcripts with timestamp handling
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "(?<=\\[[0-9:]+\\])\\s", "(?<=[.!?])\\s+", " "]
    )
    
    # Split transcript into chunks
    chunks = text_splitter.split_text(transcript_text)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_texts(chunks, embeddings)
    
    return {
        "vectorstore": vectorstore,
        "chunk_count": len(chunks),
        "sample_chunk": chunks[0][:100] if chunks else ""
    }



def extract_youtube_video_id(url: str) -> str:
    """
    Extracts the video ID from a YouTube URL.
    
    Supports standard youtube.com URLs, youtu.be short URLs, 
    and embedded URLs.
    
    Args:
        url: YouTube URL
        
    Returns:
        YouTube video ID or empty string if not found
    """
    # Handle multiple URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/embed\?v=)([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return ""


def extract_trade_date_from_video(video_id: str) -> dict:
    """
    Extract the trade date based on the video's upload date.
    Assumes the trade happened on the same day the video was uploaded.

    Args:
        video_id: YouTube video ID

    Returns:
        Dictionary with date information
    """
    try:
        # Get YouTube API key from environment
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            return {
                "detected_dates": [],
                "most_likely_date": None,
                "confidence_score": 0,
                "error": "YouTube API key not found in environment variables"
            }

        # Initialize YouTube API client
        youtube = build("youtube", "v3", developerKey=api_key)

        # Get video details
        video_response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        # Extract upload date
        if not video_response.get("items"):
            return {
                "detected_dates": [],
                "most_likely_date": None,
                "confidence_score": 0,
                "error": f"No video found with ID: {video_id}"
            }

        upload_date_str = video_response["items"][0]["snippet"]["publishedAt"]

        # Convert to datetime and format as readable date
        upload_datetime = datetime.strptime(upload_date_str, "%Y-%m-%dT%H:%M:%S%z")
        formatted_date = upload_datetime.strftime("%Y-%m-%d")
        readable_date = upload_datetime.strftime("%B %d, %Y")

        return {
            "detected_dates": [readable_date],
            "most_likely_date": formatted_date,  # YYYY-MM-DD format for API calls
            "readable_date": readable_date,      # Human-readable format
            "confidence_score": 90,              # High confidence since it's from metadata
            "source": "video_upload_date"
        }

    except Exception as e:
        return {
            "detected_dates": [],
            "most_likely_date": None,
            "confidence_score": 0,
            "error": f"Error extracting upload date: {str(e)}"
        }


def extract_trading_summary(vectorstore, ticker): 
    """
    Extract details about the trade that took place like entry/exit but
    more of an overview since videos aren't likely to state explicit entry/exit.
    Args:
        vectorstore: vector store for semantic search
        ticker: stock ticker symbol
    returns:
        Dictionary with trade summary
    """
    summary_queries = [
        f"Summarize the {ticker} trade",
        f"what was the setup for the {ticker} trade",
        f"how did the {ticker} trade unfold",
        f"describe the entry and exit for {ticker}",
        f"what strategy was used on {ticker}",
        f"what was the though process behind the {ticker} trade"
    ]
    summary_results = []
    for query in summary_queries:
        results = vectorstore.similarity_search(query, k=3)
        summary_results.extend([r.page_content for r in results])

    setup = extract_component(summary_results, ["setup", "pattern", "identified", "looking at"])
    execution = extract_component(summary_results, ["bought", "entered", "sold", "exited"])
    outcome = extract_component(summary_results, ["profit", "gain", "loss", "result"])
    lessons = extract_component(summary_results, ["learned", "mistake", "improvement", "next time"])

    # compile most relevant segments
    return {
        "setup" : setup,
        "execution": execution,
        "outcome": outcome,
        "lessons": lessons,
        "full_summary": "\n".join(summary_results[:5])
    }

# component extract helper function
def extract_component(results, keywords):
    """Extract relevant segments containing keywords"""
    relevant_segments = []
    for result in results:
        sentences = re.split(r'(?<=[.!?])\s+', result)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                relevant_segments.append(sentence)

    return relevant_segments[:3]

def extract_ticker_info(transcript_text, vectorstore):
    """
    Extracts stock ticker symbols from transcript with confidence scoring.
    
    Uses both regex pattern matching and semantic search for higher accuracy.
    
    Args:
        transcript_text: Full transcript text
        vectorstore: Vector store for semantic search
        
    Returns:
        Dictionary with ticker information and confidence scores
    """
    # Pattern-based extraction
    ticker_pattern = r'\$[A-Z]{1,5}|(?<!\w)[A-Z]{2,5}(?!\w)'
    matches = re.findall(ticker_pattern, transcript_text)
    
    # Filter false positives (common abbreviations and words)
    false_positives = {'CEO', 'CFO', 'CTO', 'COO', 'I', 'A', 'THE', 'ON', 'IN', 'IS', 'AM', 'PM', 'ET', 
                       'PDT', 'EST', 'PST', 'MST', 'GMT', 'UTC', 'AND', 'BUT', 'OR', 'IF', 'HOW', 'WHY', 
                       'WHAT', 'WHEN', 'WHO', 'WHICH', 'ANY', 'ALL', 'FOR', 'TO', 'BY', 'MACD', 'VWAP', 'SMAS', 'SMA'}
    
    filtered_matches = [m for m in matches if m.replace('$', '') not in false_positives]
    
    # Count occurrences
    ticker_counts = {}
    for ticker in filtered_matches:
        clean_ticker = ticker.replace('$', '')
        ticker_counts[clean_ticker] = ticker_counts.get(clean_ticker, 0) + 1
    
    # Sort by frequency
    sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Semantic search for ticker confirmation
    ticker_queries = [
        "stock ticker symbol being traded",
        "which stock was traded in this video",
        "the ticker symbol of the stock in this trade"
    ]
    
    semantic_results = []
    for query in ticker_queries:
        results = vectorstore.similarity_search(query, k=2)
        semantic_results.extend([r.page_content for r in results])
    
    # Calculate confidence score
    confidence_score = 0
    most_likely_ticker = None
    if sorted_tickers:
        most_likely_ticker = sorted_tickers[0][0]
        count = sorted_tickers[0][1]
        
        # Base confidence from frequency
        confidence_score = min(90, count * 15)  # Cap at 90%
        
        # Boost if ticker appears in semantic results
        if most_likely_ticker in ' '.join(semantic_results):
            confidence_score += 10
            
        # Cap at 100%
        confidence_score = min(100, confidence_score)
    
    return {
        "detected_tickers": [{"ticker": t, "count": c} for t, c in sorted_tickers],
        "most_likely_ticker": most_likely_ticker,
        "confidence_score": confidence_score,
        "semantic_context": semantic_results[:2]  # Provide context snippets
    }



def analyze_trading_transcript(transcript_text: str, url: str) -> str:
    """
    Performs comprehensive semantic analysis of a trading video transcript to extract
    structured trade information.
    
    Args:
        transcript_text: Full transcript text with timestamps
        url: youtube video url
        
    Returns:
        JSON string containing structured trade information
    """
    try:
        if not transcript_text or transcript_text.startswith("Error"):
            return transcript_text  # Return error message if transcript retrieval failed
        
        # Create vector store from transcript
        vector_data = create_vector_store(transcript_text)
        vectorstore = vector_data["vectorstore"]
        
        # Extract ticker symbol with confidence scoring
        ticker_info = extract_ticker_info(transcript_text, vectorstore)
       
        video_id = extract_youtube_video_id(url)
        # Extract trade date information
        #date_info = extract_trade_date(vectorstore)
        date_info = extract_trade_date_from_video(video_id)
        
        # commented out the old entry/exit functions to call the 
        summary_info = extract_trading_summary(vectorstore, ticker_info["most_likely_ticker"])
        # Extract trading strategy
                
        # Compile all information into structured format
        trade_analysis = {
            "ticker": ticker_info,
            "date": date_info,
            "summary": summary_info
        }
        
        return json.dumps(trade_analysis, indent=2)
        
    except Exception as e:
        return f"Error analyzing trading transcript: {str(e)}"



@tool
def analyze_trade_video_from_url(youtube_url: str) -> str:
    """
    Complete end-to-end analysis of a trading video from URL.
    Gets transcript, extracts trade details, and verifies against market data.
    
    Args:
        youtube_url: Full YouTube URL
        
    Returns:
        JSON string with complete trade analysis and verification
    """
    try:
        # Step 1: Get video transcript
        transcript = get_youtube_transcript_from_url(youtube_url)
        if transcript.startswith("Error"):
            return json.dumps({
                "success": False,
                "error": transcript,
                "stage": "transcript_retrieval"
            })
        
        # Extract video ID for date retrieval
        video_id = extract_youtube_video_id(youtube_url)
        if not video_id:
            return json.dumps({
                "success": False,
                "error": f"Could not extract video ID from URL: {youtube_url}",
                "stage": "video_id_extraction"
            })
        
        # Step 2: Get video upload date for trade date
        date_info = extract_trade_date_from_video(video_id)
        trade_date = date_info.get("most_likely_date")
        
        if not trade_date:
            return json.dumps({
                "success": False,
                "error": date_info.get("error", "Could not determine trade date"),
                "stage": "date_extraction"
            })
        
        # Step 3: Analyze transcript for trade details
        analysis = analyze_trading_transcript(transcript, youtube_url)
        if analysis.startswith("Error"):
            return json.dumps({
                "success": False,
                "error": analysis,
                "stage": "transcript_analysis"
            })
        
        analysis_data = json.loads(analysis)
        
        # Extract key information
        ticker = analysis_data["ticker"]["detected_tickers"]
        
        if not ticker:
            return json.dumps({
                "success": False,
                "error": "Could not identify ticker symbol from transcript",
                "stage": "ticker_identification",
                "analysis": analysis_data
            })
        
        # Step 5: Compile final results
        return json.dumps({
            "success": True,
            "video_url": youtube_url,
            "video_id": video_id,
            "trade_date": trade_date,
            "tickers": ticker,
            "trade_analysis": analysis_data
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "stage": "unknown"
        })

    