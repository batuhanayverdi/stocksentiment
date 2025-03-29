📈 Integrated Stock Sentiment App (InvestingAI-6)
This is a Streamlit-based web application that combines financial news sentiment analysis with technical indicators to support stock market investment decisions. The project was developed as part of the Campus Challenge: Successful Investing with AI and Large Language Models at the Technical University of Munich.

💡 Key Features
🟢 Live Stock Ticker
Continuously updated prices for a wide set of S&P 500 stocks.

📊 Technical Analysis Indicators

RSI (Relative Strength Index)

Momentum

Sharpe Ratio using GARCH(1,1) volatility modeling

📰 Sentiment Analysis
Real-time financial news is fetched via Yahoo Finance RSS feeds and analyzed using FinBERT (ProsusAI/finbert) to classify sentiment as Positive, Neutral, or Negative.

🤖 Hybrid Investment Decision Engine
A decision-making function combines RSI, Momentum, Sharpe Ratio, and sentiment scores (with double weight) to suggest BUY, SELL, or HOLD.

📥 Excel Report Export
Automatically generates an Excel report with article sentiment and color-coded rows.

🚀 How to Run
1. Clone the Repository
bash
Kopyala
Düzenle
git clone https://github.com/yourusername/investingai-app.git
cd investingai-app
2. Install Requirements
bash
Kopyala
Düzenle
pip install -r requirements.txt
3. Run the App
bash
Kopyala
Düzenle
streamlit run main2.py
🔧 Tools & Libraries
Streamlit – for interactive web interface

Yahoo Finance API / Alpha Vantage – for financial data

Transformers (HuggingFace) – for FinBERT sentiment model

Plotly – for beautiful data visualizations

yfinance, pandas, openpyxl, arch – for financial modeling and Excel export

📈 Strategy Background
This application supports a hybrid investment strategy, inspired by your presentation:

Fundamental & Technical Analysis: Includes traditional indicators (Sharpe Ratio, RSI, Momentum)

ML Integration: Used FinBERT to process news sentiment

Weekly Retraining: Strategy can be adapted with updated data and insights

📤 Outputs
Interactive investment recommendations

Visual stock performance & sentiment distribution

Downloadable sentiment analysis report in .xlsx format

📚 Related Presentation
Machine Learning in Investing: A Hybrid Strategy
Batuhan Ayverdi & Hamide Nur Tutuk
Technical University of Munich
