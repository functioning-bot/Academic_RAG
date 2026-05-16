![International Journal for Novel Research in Economics, Finance and Management logo](page_1_image_1_v2.jpg)

# A Multi-Agent Deep Learning and Reinforcement Learning Framework for Stock Price Prediction and Buy–Sell Decision Support in Volatile Equity Markets

GLN Sravan Kumar¹, Dr. Kasireddy Sandeep Reddy², K. Santoshini³, B. Ramesh⁴

¹'²Assistant Professor, Department of Business Management, Malla Reddy College of Engineering & Technology, Maisammaguda, Secunderabad.

³Assistant Professor, Malla Reddy College of Engineering & Technology, Maisammaguda, Secunderabad

⁴Assistant Professor, Department of Business Management, Sri Indu Institute of Engineering & Technology, Sheriguda (V), Ibrahimpatnam (M), R.R. District

Abstract – This study proposes a multi-agent deep learning framework, termed Neural-Arbitrage, designed to enhance stock price prediction and buy–sell decision support in volatile equity markets. The framework integrates convolutional neural networks, recurrent neural architectures, and deep reinforcement learning to model non-linear price dynamics, volatility clustering, and regime shifts. Using historical secondary market data for Apple Inc. (AAPL) and Tesla Inc. (TSLA), representing contrasting volatility profiles, the proposed model is empirically evaluated against traditional econometric models (ARIMA) and single-agent neural models (LSTM). Model performance is assessed using mean squared error, directional accuracy, and Sharpe ratio. Results indicate that the proposed multi-agent architecture achieves superior predictive accuracy and significantly improves risk-adjusted returns, particularly during high-volatility periods. The findings demonstrate the effectiveness of cooperative learning and reinforcement-driven decision optimization in dynamic financial environments, contributing to the growing literature on intelligent algorithmic trading systems.

Keywords – Multi-Agent Deep Learning, Algorithmic Trading, Reinforcement Learning, Stock Price Prediction, Volatility Modeling, Risk-Adjusted Returns.

## I. INTRODUCTION

The contemporary equity market is marked by more uncertainty, and the pace of information dissemination is increasing sharply, which puts major restrictions on the traditional quantitative modeling methods. Within a setting of high volatility, it is the asymmetric information dispersion, company-specific shocks, and changing behavior of investors that also influence the price of the assets in addition to the underlying fundamental evaluations. Such circumstances demonstrate that many existing econometric frameworks fail to capture the complexity and non-linear nature of interactions which exist in high-frequency financial data, which also requires more adaptive and data-driven analytical approaches.

This research fills this gap by proposing a multi-agent deep learning (MADL) framework that can help to improve the estimation of return dynamics and underlying volatility structures. The suggested Neural- Arbitrage model combines heterogeneous data streams, such as technical indicators, sentiment-based signals, and macroeconomic inputs, with the help of an integrated learning setting. The system produces both localized market behavior forecasts and overall liquidity situation forecasts by tasking specialized agents (sector-focused evaluators and risk-monitoring components) with the analysis of the market and, consequently, forecasting the market-wide and macroeconomic phenomena. With such a collaborative structure, predictions can be made more resilient through the fact that risk awareness is incorporated into the actual learning process.

Besides predictive modeling, the study goes further to support trade execution with deep reinforcement learning (DRL) to enable actionable decision-making. Instead of using predetermined signals that are encoded by set rules, the DRL component keeps interacting with the market environment in order to update its decision policies. The system employs a Markov Decision Process to model trading behavior through the use of an iterative process that is used to adapt the strategies to maximize performance goals (i.e. profitability and risk-adjusted returns). This adaptive learning enables the framework to react to sudden market shifts, both severe declines and swift surges, without relying on a set of heuristics.

The empirical validity of the suggested approach is discussed on the basis of the comparative analysis of Apple Inc. (AAPL) and Tesla Inc. (TSLA) that represent opposite profiles of equity in the stability and volatility. AAPL can be mentioned as an example of a mature institutionally preferred stock, whose price dynamics are relatively stable, whereas TSLA is characterized by a strong dependence on the market mood and the tendencies to speculation. The findings show that, as effective Long Short-Term Memory models may be in terms of capturing long-term temporal dynamics, more adaptable hybrid models are needed to address ad hoc regime shifts that usually occur with technology-driven equities. The analysis of the disparities in predictive performance and risk-adjusted performance between these assets confirms the ability of intelligent systems to be adapted to a wide range of volatility environments, which supports the importance of advanced AI models in modern financial decision-making.

---

**Research Gap and Contribution**
Despite extensive research on deep learning–based stock prediction models, existing studies largely rely on single-agent architectures and static decision rules, limiting adaptability during abrupt market regime shifts. Furthermore, limited integration exists between predictive intelligence and execution-oriented reinforcement learning under realistic risk constraints. This study addresses these gaps by proposing a cooperative multi-agent learning framework that explicitly integrates volatility awareness and deep reinforcement learning for decision optimization. The key contribution lies in demonstrating how agent-level specialization and coordinated learning improve both predictive robustness and risk-adjusted trading performance in volatile equity markets.

# II. LITERATURE REVIEW

**Comparative Analysis of Neural Architectures**
**Nichani, Gasmi, et al. (2025):** Performed a ground-breaking comparative experiment between LSTM and Transformer models of high-frequency financial time series. They report that although LSTMs are more suited to predicting stable and long-term trends in equities such as Microsoft (MSFT) and Transformers are more receptive to detecting so-called volatility spikes in assets with more volatile behavior such as Nvidia (NVDA), because of their self-attention mechanism.

**Alharbi, M. (2024):** Assessed the forecasting performance of deep learning on world markets (S&P 500, DAX). This study has determined that CNN-LSTM hybrids offer a unique edge in explaining unexpected changes in regime by considering price-volume data as multi-dimensional features, lowering the Mean Absolute Percentage Error (MAPE) to less than $3\%$ in high-liquidity markets.

**Tanaka, Fujimoto, & Sakamoto (2025):** The first to incorporate Adaptive Risk Metrics (VaR and CVaR) on the loss functions of Transformer models. This risk-aware forecasting method enables both price direction-based buy-sell indicators and also the likelihood of downside tail-risk, to offer a more reliable institutional portfolio management framework.

**Kong, Wang, & Wen (2025):** Investigated the "Memory Capacity" of neural networks, introducing the sLSTM (Stabilized LSTM). Their work addresses the vanishing gradient problem in long-sequence forecasting, proving that exponentially-weighted forget gates can enhance the model's ability to recall "market shocks" from previous cycles (e.g., the 2020 pandemic) to better inform 2025 buy-sell decisions.

**Sangve, Kohad, et al. (2025):** Developed the "ProfitPulse" framework, which utilizes Deep Q-Networks (DQN) for reinforcement learning-driven trading. By simulating realistic capital constraints and transaction costs, the authors demonstrated that AI agents can consistently outperform traditional "Buy-and-Hold" strategies by dynamically rebalancing positions in response to real-time sentiment scraped from financial news.

**Critical Synthesis of Prior Studies**
While prior research demonstrates the effectiveness of LSTM and Transformer-based models in capturing temporal dependencies and volatility spikes, these approaches predominantly focus on isolated prediction tasks without incorporating decision optimization mechanisms. Risk-aware models integrating VaR or CVaR metrics improve downside protection but remain computationally constrained within single-agent frameworks. Reinforcement learning–based trading systems, though effective in dynamic allocation, often depend on externally generated signals rather than integrated predictive agents. Consequently, there remains a methodological gap in unifying multi-agent predictive intelligence with reinforcement-driven execution under volatility-sensitive environments. The proposed Neural Arbitrage framework seeks to bridge this gap through cooperative agent learning and integrated decision optimization.

**A Multi-Agent Systems in Finance**
Multi-agent learning frameworks allocate specialized predictive agents (e.g., trend agents, volatility agents) whose combined signals aim to improve system robustness during market regime changes.

**Objectives of the Study**
* To design a Neural-Arbitrage–based multi-agent deep learning framework for forecasting stock price movements in volatile equity markets.
* To integrate convolutional neural networks, recurrent neural networks, and deep reinforcement learning to capture non-linear patterns, volatility clustering, and regime shifts in financial time-series data.
* To evaluate the effectiveness of cooperative multi-agent learning in improving buy–sell decision support under varying market liquidity and stress conditions.
* To compare the predictive accuracy and directional reliability of the proposed framework with traditional econometric models and single-agent neural models.
* To assess the framework’s ability to enhance risk-adjusted performance using metrics such as Mean Squared Error, Directional Accuracy, Sharpe Ratio, and excess returns.
* To validate the practical applicability of the proposed model through empirical analysis using real-world equity data of Apple Inc. (AAPL) and Tesla Inc. (TSLA), representing contrasting volatility regimes.

# III. METHODOLOGY

**Research Design**
This study adopts a quantitative, experimental research design to develop and evaluate a multi-agent deep learning and reinforcement learning framework for stock price

---

prediction and buy–sell decision support under volatile market conditions. The proposed Neural-Arbitrage framework is empirically validated using historical secondary market data and benchmarked against traditional econometric and single-agent neural models.

## Data Sources and Preprocessing
Daily historical data for Apple Inc. (AAPL) and Tesla Inc. (TSLA) were collected for a three-year period and include:
* Open, High, Low, Close (OHLC) prices
* Trading volume
* Historical volatility measures
* Derived technical indicators (RSI, MACD, moving averages)

To prevent look-ahead bias, all features were strictly lagged and constructed using only information available up to time t. Data preprocessing involved:
* Min–max normalization
* Sliding-window segmentation
* Noise filtering
* Feature augmentation

The dataset was divided using a rolling-window approach into training (70%), validation (15%), and testing (15%) sets.

## Neural-Arbitrage Multi-Agent Architecture
The proposed framework consists of four interacting agent types:

### Feature Extraction Layer
Convolutional Neural Networks (CNNs) are employed to capture local price-volume patterns, while Recurrent Neural Networks (LSTM) model long-term temporal dependencies and regime persistence.

### Prediction Agents
Multiple prediction agents operate over heterogeneous temporal horizons:
* Short-term (10–20 days)
* Medium-term (30–60 days)
* Long-term (90–180 days)

### Each agent produces forecasts for:
* Price direction
* Expected returns
* Trend strength

This agent-level specialization enhances robustness under regime shifts.

### Volatility Agent
A dedicated volatility agent estimates realized volatility using rolling standard deviation and volatility clustering patterns. The volatility forecasts serve as risk-awareness signals and are integrated into the decision-making process to dynamically adjust trading aggressiveness.

### Reinforcement Learning–Based Decision Support
The trading strategy is modeled as a Markov Decision Process (MDP) defined as follows:
* **State (S<sub>t</sub>):**
Combined vector of predicted returns, volatility estimates, technical indicators, and current portfolio position.
* **Action (A<sub>t</sub>):**
{Buy, Sell, Hold}
**Reward (R<sub>t</sub>):**
$$ R_t = \Delta P_t - \lambda . \sigma_t - C_t $$
where
$\Delta P_t$ = portfolio profit,
$\sigma_t$ = predicted volatility,
$C_t$ = transaction cost,
$\lambda$ = risk-penalty coefficient.

A Deep Q-Network (DQN) is trained to learn the optimal policy that maximizes cumulative risk-adjusted returns under capital and transaction cost constraints.

### Benchmark Models
To ensure fair comparison, the Neural-Arbitrage framework is evaluated against:
* Autoregressive Integrated Moving Average (ARIMA)
* Single-agent Long Short-Term Memory (LSTM)
All models were trained using identical data splits and evaluation windows.

### Performance Evaluation Metrics
Model performance is assessed using:
* Mean Squared Error (MSE)
* Directional Accuracy
* Sharpe Ratio

These metrics jointly evaluate forecast precision, decision correctness, and risk-adjusted profitability.

### Experimental Design and Model Training
The dataset consists of daily historical price, volume, and volatility data for AAPL and TSLA covering a three-year period. Data were divided into training (70%), validation (15%), and testing (15%) sets using a rolling-window approach to prevent look-ahead bias. Neural networks were trained using the Adam optimizer with early stopping to prevent overfitting. The reinforcement learning environment incorporates proportional transaction costs and capital constraints to simulate realistic trading conditions. All experiments were conducted using consistent hyperparameters across benchmark and proposed models to ensure fair comparison.

# IV. EMPIRICAL ANALYSIS

## Data Description and Experimental Setup
The empirical analysis is conducted using daily historical data for Apple Inc. (AAPL) and Tesla Inc. (TSLA) over a three-year period. These equities are selected to represent contrasting volatility regimes, with AAPL exhibiting relatively stable price dynamics and TSLA characterized by high volatility and speculative behavior. The dataset

---

![International Journal for Novel Research in Economics, Finance and Management logo](page_4_image_2_v2.jpg)

includes price, volume, and volatility indicators, which are preprocessed using normalization and rolling-window techniques. The data are split into training (70%), validation (15%), and testing (15%) sets using a rolling-window approach to avoid look-ahead bias.

# Forecasting Performance Evaluation

Table 1: Forecasting Accuracy Comparison


| Model            | Asset | MSE ↓ | MAE ↓ | Directional Accuracy (%) ↑ |
| ---------------- | ----- | ----- | ----- | -------------------------- |
| ARIMA            | AAPL  | 0.019 | 0.103 | 54.1                       |
| LSTM             | AAPL  | 0.015 | 0.091 | 63.4                       |
| Neural-Arbitrage | AAPL  | 0.01  | 0.072 | 71.2                       |
| ARIMA            | TSLA  | 0.028 | 0.132 | 52.7                       |
| LSTM             | TSLA  | 0.021 | 0.118 | 60.3                       |
| Neural-Arbitrage | TSLA  | 0.013 | 0.081 | 74.8                       |



The Neural-Arbitrage framework consistently outperforms benchmark models across both assets. The improvement is more pronounced for TSLA, indicating superior capability in capturing non-linear price movements under high volatility. This confirms the effectiveness of the multi-agent learning architecture in enhancing predictive accuracy and directional reliability.

# Risk-Adjusted Trading Performance

Table 2: Trading Strategy Performance


| Model            | Asset | Sharpe Ratio ↑ | Excess Return (%) |
| ---------------- | ----- | -------------- | ----------------- |
| Buy & Hold       | AAPL  | 0.72           | 6.4               |
| LSTM Strategy    | AAPL  | 0.94           | 9.1               |
| Neural-Arbitrage | AAPL  | 1.18           | 13.6              |
| Buy & Hold       | TSLA  | 0.51           | 11.3              |
| LSTM Strategy    | TSLA  | 0.77           | 15.2              |
| Neural-Arbitrage | TSLA  | 1.38           | 22.9              |



The proposed framework delivers significantly higher Sharpe ratios and excess returns, particularly for TSLA. This demonstrates that integrating deep reinforcement learning with predictive agents leads to economically meaningful and risk-adjusted trading decisions.

# Volatility Regime Analysis

Table 3: Performance Under Different Volatility Regimes (TSLA)


| Model            | Low Volatility Sharpe | High Volatility Sharpe |
| ---------------- | --------------------- | ---------------------- |
| Buy & Hold       | 0.62                  | 0.41                   |
| ARIMA Strategy   | 0.68                  | 0.49                   |
| LSTM Strategy    | 0.91                  | 0.77                   |
| Neural-Arbitrage | 1.12                  | 1.38                   |



While all models perform comparably during low-volatility periods, Neural-Arbitrage exhibits a substantial advantage during high-volatility regimes. This validates the role of the volatility agent and cooperative learning in adapting to market stress conditions.

# Agent Contribution Analysis

Table 4: Ablation Study


| Configuration                 | MSE ↓ | Directional Accuracy (%) ↑ | Sharpe Ratio ↑ |
| ----------------------------- | ----- | -------------------------- | -------------- |
| Prediction Agent only         | 0.017 | 62.1                       | 0.83           |
| Prediction + Volatility Agent | 0.014 | 67.5                       | 1.02           |
| Prediction + DRL Agent        | 0.013 | 69.8                       | 1.14           |
| Full Neural-Arbitrage         | 0.01  | 74.3                       | 1.28           |



Each agent contributes incrementally to performance improvement. The full Neural-Arbitrage architecture achieves the best results, confirming that cooperative multi-agent learning enhances both forecasting accuracy and trading effectiveness.

# Robustness Analysis

Table 5: Rolling-Window Performance Stability


| Period | Sharpe Ratio | Return Volatility |
| ------ | ------------ | ----------------- |
| Year 1 | 1.19         | 0.021             |
| Year 2 | 1.32         | 0.018             |
| Year 3 | 1.28         | 0.019             |



Stable performance across rolling windows indicates robustness and consistency of the proposed framework, supporting its practical applicability in real-world trading environments.


| Metric           | Value                   |
| ---------------- | ----------------------- |
| Apple Inc (AAPL) | $273.81                 |
| Change           | +$1.52 (+0.56%)         |
| Date             | December 24             |
| After Hours      | $272.27 -$1.54 (-0.56%) |
| Open             | 272.26                  |
| Day Low          | 271.31                  |
| Day High         | 275.41                  |
| Volume           | 17.4M                   |
| 52 Week Low      | 169.23                  |
| 52 Week High     | 288.42                  |
| Market Cap (TTM) | 3.81T                   |
| EPS (TTM)        | 6.59                    |
| P/E Ratio (TTM)  | 30.26                   |



Figure 1: Actual vs. Predicted Stock Prices: Neural-Arbitrage Model Performance (TSLA)

**Company Selection & Market Context** This study uses real market data for comparison:

---

| Time     | Price (USD) |
| -------- | ----------- |
| 10:00 AM | 485.40      |
| 11:00 AM | 488.30      |
| 12:00 PM | 476.89      |
| 1:00 PM  | 490.68      |



Figure 2: Actual vs. Predicted Stock Prices: Neural- Arbitrage Model Performance (AAPL)

Apple Inc (AAPL): A large-cap technology firm with moderated volatility and higher fundamental metrics.

### Future Scope of the Study

Future research can extend the Neural-Arbitrage framework to multi-asset portfolios, including commodities, cryptocurrencies, and fixed-income securities, to examine cross-market applicability. The integration of real-time alternative data sources such as social media sentiment, earnings call transcripts, and macroeconomic news may further enhance predictive accuracy. Incorporating explainable AI techniques can improve model transparency and regulatory acceptance. Additionally, risk-aware reinforcement learning using downside risk measures such as Conditional Value at Risk (CVaR) can strengthen portfolio protection. Real-time deployment and live-market testing under varying liquidity conditions would further validate the framework’s practical effectiveness.

# V. CONCLUSION

This paper introduces Neural-Arbitrage as a strong, adjustable, and empirically tested multi-agent deep learning platform that has the power to deal with the existing complexities of high-volatility equity markets. The proposed architecture allows overcoming the limits of the non-linear dependence, volatile groups of volatility, and regime shifts characteristic of the contemporary financial market by overcoming the constraints of the fixed econometric and single-agent neural models. The results of the AAPL and TSLA empirical studies indicate a statistically significant enhancement in the predictive accuracy, directional reliability and the risk-adjusted performance, which proves the effectiveness of multi-agent cooperation and deep reinforcement learning in financial decision-making.

In addition, critical connection between predictive intelligence and action execution has been defined in the research which puts Neural- Arbitrage at a distinctive position not only as a predictive tool but as a decision support system. The high Sharpe ratios, and lower number of false signals generated by the framework in volatile times highlights the practical applicability of the framework to institutional investors, quantitative funds, as well as policy-focused financial analysts. This research reaffirms the strategic need to have smart, dynamic, and risk-sensitive trading platforms in an age of informational asymmetry and algorithmic competition.

### Recommendations

Practically, it is recommended that financial institutions and algorithmic trading desks should implement multi-agent AI architectures where the analytical responsibility of both specialized agents is distributed as opposed to using monolithic prediction models. The systems also improve the strength in the face of market stress, as well as exposure to model overfitting.

Regulators and market supervisors ought to take into consideration the development of AI governance frameworks to enhance transparency, explainability, and ethical usage of the autonomous trading systems as far as policy and governance are concerned. To preserve systemic stability in more automated financial markets, the standardization of stress-testing procedures on AI-based trading systems will be necessary.

### REFERENCES

1. Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. Journal of Finance, 25(2), 383–417.

2. Merton, R. C. (1973). Theory of rational option pricing. Bell Journal of Economics, 4(1), 141–183.

3. Hochreiter, S., & Schmidhuber, J. (1997). Long short- term memory. Neural Computation, 9(8), 1735–1780.

4. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.

5. Goodfellow, I., Bengio, Y., & Courville, (2016). Deep Learning. MIT Press.

6. Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance, 1(2), 223–236.

7. Tsay, R. S. (2010). Analysis of Financial Time Series. Wiley.

8. LeBaron, B. (2006). Agent-based computational finance. Handbook of Computational Economics, 2, 1187–1233.

9. Moody, J., & Saffell, M. (2001). Learning to trade via reinforcement learning. IEEE Transactions on Neural Networks, 12(4), 875–889.

10. Fischer, T., & Krauss, C. (2018). Deep learning with LSTM networks for financial market predictions. European Journal of Operational Research, 270(2), 654–669.

11. Heaton, J., Polson, N., & Witte, J. (2017). Deep learning in finance. Annual Review of Financial Economics, 9, 145–181.

12. Zhang, Y., Aggarwal, C., & Qi, G. J. (2017). Stock price prediction via discovering multi-frequency trading patterns. KDD.

---

13. Jiang, Z., Xu, D., & Liang, J. (2017). A deep reinforcement learning framework for the financial portfolio management problem. arXiv preprint.

14. Chen, Y., & Hao, Y. (2017). A feature weighted support vector machine for stock prediction. Expert Systems with Applications, 80, 340–347.

15. Kim, K. J. (2003). Financial time series forecasting using support vector machines. Neurocomputing, 55(1–2), 307–319.

16. Lo, A. W. (2004). The adaptive markets hypothesis. Journal of Portfolio Management, 30(5), 15–29.

17. Engle, R. F. (2001). GARCH 101.

18. Journal of Economic Perspectives, 15(4), 157–168.

19. Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. Journal of Computational Science, 2(1), 1–8.

20. Kearns, M., & Nevmyvaka, Y. (2013). Machine learning for market microstructure. High Frequency Trading.

21. Silver, D., et al. (2016). Mastering the game of Go with deep neural networks. Nature, 529, 484–489.

22. Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2016). Deep direct reinforcement learning for financial signal representation. IEEE TNNLS, 28(3), 653–664.

23. Shiller, R. J. (2003). From efficient markets theory to behavioral finance. Journal of Economic Perspectives, 17(1), 83–104.

24. Poterba, J. M., & Summers, L. H. (1988). Mean reversion in stock prices. Journal of Financial Economics, 22(1), 27–59.

25. Zhang, S., et al. (2020). Deep learning based financial time series forecasting. IEEE Access, 8, 113583–113596.

26. Alharbi, M. (2024). Deep neural networks for market regime detection. Expert Systems with Applications.

27. Tanaka, H., Fujimoto, K., & Sakamoto, Y. (2025). Risk-aware transformer models for trading. Quantitative Finance.

28. Kong, X., Wang, Y., & Wen, F. (2025).

29. Stabilized LSTM for long-horizon market forecasting. Neural Networks.

30. Sangve, S., Kohad, R., et al. (2025). Reinforcement learning-based algorithmic trading systems. Journal of Financial Data Science.