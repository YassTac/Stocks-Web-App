from flask import Flask, render_template, request, jsonify
import pandas as pd
import os


def load_tickers(path: str = "ticker_biotech.csv") -> pd.DataFrame:
	df = pd.read_csv(path, header=0)
	# Clean column names and values (strip quotes/spaces)
	df.columns = [c.strip().strip('"') for c in df.columns]
	if 'ticker' in df.columns:
		df['ticker'] = df['ticker'].astype(str).str.strip().str.strip('"')
	return df


app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
	df = load_tickers()
	tickers = df['ticker'].dropna().astype(str).unique().tolist()
	return render_template('index.html', tickers=tickers)


@app.route('/price')
def price():
	ticker = request.args.get('ticker')
	period = request.args.get('period', '1y')
	if not ticker:
		return jsonify({'error': 'ticker is required'}), 400
	try:
		import yfinance as yf
	except Exception:
		return jsonify({'error': 'yfinance not installed: pip install yfinance'}), 500

	try:
		data = yf.download(ticker, period=period, progress=False)
	except Exception as e:
		return jsonify({'error': f'failed to download data: {str(e)}'}), 500

	if data is None or data.empty:
		return jsonify({'error': 'no data found for ticker'}), 404

	data = data.reset_index()
	# Convert dates to ISO strings
	dates = data.iloc[:, 0].astype(str).tolist()
	closes = data['Close'].round(4).fillna(None).tolist()
	return jsonify({'dates': dates, 'closes': closes})


if __name__ == '__main__':
	port = int(os.environ.get('PORT', 8501))
	app.run(debug=True, port=port)


