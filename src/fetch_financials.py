import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/financials.log'),
        logging.StreamHandler()
    ]
)

# Constants
DATA_PATH = "data/"
STOCK_SYMBOL = "AAPL"
os.makedirs(DATA_PATH, exist_ok=True)

def fetch_earnings(stock_symbol=STOCK_SYMBOL):
    """
    Fetch net income and revenue from the income statement.

    Parameters:
    - stock_symbol (str): Stock ticker (e.g., "AAPL")

    Returns:
    - DataFrame containing quarterly net income and revenue.
    """
    try:
        stock = yf.Ticker(stock_symbol)

        # Extract quarterly income statement
        income_stmt = stock.quarterly_income_stmt.T
        income_stmt.index = pd.to_datetime(income_stmt.index)

        # Select key financial data
        earnings_data = income_stmt[["Total Revenue", "Net Income"]].rename(
            columns={"Total Revenue": "Revenue", "Net Income": "Earnings"}
        )

        file_path = os.path.join(DATA_PATH, "earnings.parquet")
        earnings_data.to_parquet(file_path, compression="snappy")
        logging.info(f"✅ Earnings data saved to {file_path}")
        
        # Log some statistics
        latest = earnings_data.iloc[-1]
        logging.info(f"\nLatest Quarterly Results:")
        logging.info(f"Revenue: ${latest['Revenue']/1e9:.2f}B")
        logging.info(f"Earnings: ${latest['Earnings']/1e9:.2f}B")
        
        return earnings_data

    except Exception as e:
        logging.error(f"Error fetching earnings data: {e}")
        return None

def fetch_balance_sheet(stock_symbol=STOCK_SYMBOL):
    """
    Fetch balance sheet data for a given stock.

    Parameters:
    - stock_symbol (str): Stock ticker (e.g., "AAPL")

    Returns:
    - DataFrame containing balance sheet data.
    """
    try:
        stock = yf.Ticker(stock_symbol)

        # Extract balance sheet data
        balance_sheet = stock.quarterly_balance_sheet.T
        balance_sheet.index = pd.to_datetime(balance_sheet.index)
        
        # Create a simplified balance sheet with key metrics
        simplified_bs = pd.DataFrame()
        
        # Total Assets and Liabilities
        simplified_bs['Total Assets'] = balance_sheet['Total Assets']
        simplified_bs['Total Liabilities'] = balance_sheet['Total Liabilities Net Minority Interest']
        simplified_bs['Stockholder Equity'] = balance_sheet['Stockholders Equity']
        
        # Current Assets and Liabilities
        simplified_bs['Current Assets'] = balance_sheet['Current Assets']
        simplified_bs['Current Liabilities'] = balance_sheet['Current Liabilities']
        
        # Cash and Investments
        simplified_bs['Cash'] = balance_sheet['Cash Financial']
        if 'Other Short Term Investments' in balance_sheet.columns:
            simplified_bs['Short Term Investments'] = balance_sheet['Other Short Term Investments']
        
        # Other important metrics
        simplified_bs['Receivables'] = balance_sheet['Receivables']
        simplified_bs['Inventory'] = balance_sheet['Inventory']
        simplified_bs['Long Term Debt'] = balance_sheet['Long Term Debt']
        simplified_bs['PPE'] = balance_sheet['Net PPE']

        file_path = os.path.join(DATA_PATH, "balance_sheets.parquet")
        simplified_bs.to_parquet(file_path, compression="snappy")
        logging.info(f"✅ Balance sheet data saved to {file_path}")
        
        # Log some key metrics
        latest = simplified_bs.iloc[-1]
        logging.info(f"\nLatest Balance Sheet Metrics:")
        logging.info(f"Total Assets: ${latest['Total Assets']/1e9:.2f}B")
        logging.info(f"Total Liabilities: ${latest['Total Liabilities']/1e9:.2f}B")
        logging.info(f"Stockholder Equity: ${latest['Stockholder Equity']/1e9:.2f}B")
        
        if 'Current Assets' in latest.index and 'Current Liabilities' in latest.index:
            current_ratio = latest['Current Assets'] / latest['Current Liabilities']
            logging.info(f"Current Ratio: {current_ratio:.2f}")
        
        return simplified_bs

    except Exception as e:
        logging.error(f"Error fetching balance sheet data: {e}")
        return None

def fetch_financial_ratios(stock_symbol=STOCK_SYMBOL):
    """
    Fetch key financial ratios including EPS and P/E ratio.

    Parameters:
    - stock_symbol (str): Stock ticker (e.g., "AAPL")

    Returns:
    - DataFrame containing financial ratios.
    """
    try:
        stock = yf.Ticker(stock_symbol)

        # Extract financial ratios
        info = stock.info

        ratios = {
            "Stock": stock_symbol,
            "Market Cap": info.get("marketCap"),
            "P/E Ratio": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "EPS": info.get("trailingEps"),
            "Book Value": info.get("bookValue"),
            "Revenue": info.get("totalRevenue"),
            "Net Income": info.get("netIncomeToCommon"),
            "Profit Margin": info.get("profitMargins"),
            "Operating Margin": info.get("operatingMargins"),
            "Debt/Equity": info.get("debtToEquity"),
            "ROE": info.get("returnOnEquity")
        }

        df = pd.DataFrame([ratios])
        
        # Log key ratios
        logging.info(f"\nKey Financial Ratios:")
        logging.info(f"P/E Ratio: {df['P/E Ratio'].iloc[0]:.2f}")
        logging.info(f"EPS: ${df['EPS'].iloc[0]:.2f}")
        logging.info(f"Profit Margin: {df['Profit Margin'].iloc[0]*100:.2f}%")
        logging.info(f"ROE: {df['ROE'].iloc[0]*100:.2f}%")

        file_path = os.path.join(DATA_PATH, "financial_ratios.parquet")
        df.to_parquet(file_path, compression="snappy")
        logging.info(f"✅ Financial ratios saved to {file_path}")
        
        return df

    except Exception as e:
        logging.error(f"Error fetching financial ratios: {e}")
        return None

def main():
    """Main function to fetch financial data."""
    try:
        logging.info(f"📡 Fetching financial data for {STOCK_SYMBOL}...")

        # Fetch all financial data
        earnings = fetch_earnings(STOCK_SYMBOL)
        balance_sheet = fetch_balance_sheet(STOCK_SYMBOL)
        ratios = fetch_financial_ratios(STOCK_SYMBOL)

        if all([earnings is not None, balance_sheet is not None, ratios is not None]):
            logging.info(f"\n✅ Successfully fetched all financial data for {STOCK_SYMBOL}!")
            return True
        else:
            logging.error(f"\n❌ Some financial data could not be fetched for {STOCK_SYMBOL}")
            return False

    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nFinancial data fetching interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")