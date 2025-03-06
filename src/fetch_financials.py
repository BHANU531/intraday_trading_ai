import yfinance as yf
import pandas as pd
import os

DATA_PATH = "../data/"
os.makedirs(DATA_PATH, exist_ok=True)


def fetch_earnings(stock_symbol="AAPL"):
    """
    Fetch net income and revenue from the income statement.

    Parameters:
    - stock_symbol (str): Stock ticker (e.g., "AAPL")

    Returns:
    - DataFrame containing quarterly net income and revenue.
    """
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
    print(f"✅ Earnings data saved to {file_path}")
    return earnings_data


def fetch_balance_sheet(stock_symbol="AAPL"):
    """
    Fetch balance sheet data for a given stock.

    Parameters:
    - stock_symbol (str): Stock ticker (e.g., "AAPL")

    Returns:
    - DataFrame containing balance sheet data.
    """
    stock = yf.Ticker(stock_symbol)

    # Extract balance sheet data
    balance_sheet = stock.quarterly_balance_sheet.T
    balance_sheet.index = pd.to_datetime(balance_sheet.index)

    file_path = os.path.join(DATA_PATH, "balance_sheets.parquet")
    balance_sheet.to_parquet(file_path, compression="snappy")
    print(f"✅ Balance sheet data saved to {file_path}")
    return balance_sheet


def fetch_financial_ratios(stock_symbol="AAPL"):
    """
    Fetch key financial ratios including EPS and P/E ratio.

    Parameters:
    - stock_symbol (str): Stock ticker (e.g., "AAPL")

    Returns:
    - DataFrame containing financial ratios.
    """
    stock = yf.Ticker(stock_symbol)

    # Extract financial ratios
    info = stock.info

    ratios = {
        "Stock": stock_symbol,
        "Market Cap": info.get("marketCap"),
        "P/E Ratio": info.get("trailingPE"),
        "Forward P/E Ratio": info.get("forwardPE"),
        "Earnings Per Share (EPS)": info.get("trailingEps"),
        "Book Value": info.get("bookValue"),
        "Revenue": info.get("totalRevenue"),
        "Net Income": info.get("netIncomeToCommon"),
        "Profit Margin": info.get("profitMargins"),
        "Operating Margin": info.get("operatingMargins"),
        "Debt/Equity Ratio": info.get("debtToEquity"),
        "Return on Equity (ROE)": info.get("returnOnEquity"),
    }

    df = pd.DataFrame([ratios])
    print(df)

    file_path = os.path.join(DATA_PATH, "financial_ratios.parquet")
    df.to_parquet(file_path, compression="snappy")
    print(f"✅ Financial ratios saved to {file_path}")
    return df


def main():
    """Main function to fetch financial data."""
    stock_symbol = "AAPL"  # Change this to any stock symbol

    print(f"📡 Fetching financial data for {stock_symbol}...")

    fetch_earnings(stock_symbol)
    fetch_balance_sheet(stock_symbol)
    fetch_financial_ratios(stock_symbol)

    print(f"✅ Completed fetching financial reports for {stock_symbol}!")


if __name__ == "__main__":
    main()