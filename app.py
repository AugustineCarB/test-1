
import yfinance as yf
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import dash
from dash import html, dcc

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
data = yf.download(tickers, start="2023-01-01")['Adj Close']
returns = data.pct_change().dropna()

scaled_returns = StandardScaler().fit_transform(returns.T)
pca = PCA(n_components=2).fit_transform(scaled_returns)
clusters = KMeans(n_clusters=2).fit_predict(pca)

app = dash.Dash(__name__)
fig = px.scatter(x=pca[:, 0], y=pca[:, 1], color=clusters.astype(str), hover_name=tickers)

app.layout = html.Div([
    html.H1("Portfolio PCA + Clustering"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8080)
