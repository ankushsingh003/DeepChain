import os
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from .advisor import StrategyAdvisor

logger = logging.getLogger(__name__)

class MarketStrategyAdvisor(StrategyAdvisor):
    """
    Advanced strategist that analyzes live market data to select and 
    combine strategies from the knowledge base.
    """
    
    def get_live_data(self, symbol: str, period: str = "60d", interval: str = "1d") -> pd.DataFrame:
        """Fetch real-time OHLC data via yfinance."""
        logger.info(f"Fetching live data for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    async def analyze_and_build(self, symbol: str) -> Dict[str, Any]:
        """
        The full pipeline: 
        1. Fetch Data -> 2. Analyze Market -> 3. Retrieve Strategies -> 4. Build Code -> 5. Return Plan
        """
        # 1. Fetch live data
        df = self.get_live_data(symbol)
        
        # Create a market summary for the AI
        market_summary = {
            "current_price": df['Close'].iloc[-1],
            "30d_high": df['High'].iloc[-30:].max(),
            "30d_low": df['Low'].iloc[-30:].min(),
            "avg_volume": df['Volume'].iloc[-30:].mean(),
            "recent_trend": "Bullish" if df['Close'].iloc[-1] > df['Close'].iloc[-30] else "Bearish",
            "volatility": df['Close'].iloc[-30:].std()
        }

        # 2. Retrieve Strategy Knowledge (using base class RAG logic)
        retriever = self._get_retriever()
        context = ""
        if retriever:
            rag_result = await retriever.query(
                question=f"Which of the 10 trading strategies is best for a {market_summary['recent_trend']} market with {market_summary['volatility']} volatility?",
                mode="hybrid"
            )
            context = rag_result.answer

        # 3. Prompt Gemini to synthesize a custom strategy based on LIVE data
        prompt = f"""
        System: You are a Lead Quant for DeepChain.
        
        Live Market Data for {symbol}:
        - {market_summary}
        
        Knowledge Base Context: {context}
        
        Task: 
        1. Analyze if one of the 10 strategies or a HYBRID combination is best for this specific market condition.
        2. Provide a report on WHY this strategy is selected for {symbol}.
        3. Provide the full Python implementation code that can be run on a dataframe.
        
        Requirements:
        - The code MUST use the signature: `def strategy_function(df):`
        - It must return the dataframe with a 'signals' column (1, -1, 0).
        - Focus on maximizing profit for the current {market_summary['recent_trend']} trend.
        """

        try:
            response = self.model.generate_content(prompt)
            report = response.text
        except Exception as e:
            logger.error(f"Generation error: {e}")
            report = f"Error generating dynamic strategy: {str(e)}"

        return {
            "symbol": symbol,
            "market_summary": market_summary,
            "dynamic_report": report,
            "retrieved_context": context
        }

if __name__ == "__main__":
    import asyncio
    async def test():
        advisor = MarketStrategyAdvisor()
        result = await advisor.analyze_and_build("TSLA")
        print(f"--- Analysis for {result['symbol']} ---")
        print(result["dynamic_report"])
    
    asyncio.run(test())
