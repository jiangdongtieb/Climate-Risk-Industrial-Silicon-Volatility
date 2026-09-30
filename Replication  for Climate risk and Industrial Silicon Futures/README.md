# Climate Risk and Industrial Silicon Futures Volatility | 气候风险与工业硅期货波动率

> Predicting realized volatility of industrial silicon futures using climate risk factors, HAR models, and variable selection methods (PLS / LASSO / BMA / Enet).
>
> 基于气候风险因子、HAR模型与变量选择方法（PLS/LASSO/BMA/Enet）预测工业硅期货已实现波动率。

## Overview | 项目简介

This project investigates whether climate risk factors — air pollution, weather conditions, and investor climate attention — can improve the prediction of realized volatility (RV) in industrial silicon futures. We extend the classic HAR model with 23 climate-related predictors and apply four variable selection methods to handle dimensionality.

本项目研究气候风险因子（空气污染、气象条件、投资者气候关注度）是否能提升工业硅期货已实现波动率（RV）的预测能力。在经典HAR模型框架下引入23个气候相关预测因子，并采用四种变量选择方法处理高维问题。

**Author | 作者**: Zhang Han (张涵)
**Institution | 单位**: Central University of Finance and Economics (中央财经大学)
**Advisor | 指导教师**: Zhen Fang (甄芳)
**Sample | 样本区间**: 2022-12-22 to 2025-12-21 (726 daily observations)

## Repository Structure | 仓库结构

```
├── code/
│   ├── har_models/              # HAR baseline and 6 extended models
│   ├── variable_selection/      # PLS, LASSO, BMA, Enet + selection results + sign stats
│   ├── data_processing/         # RV/RK calculation, weather processing, weight merge
│   ├── economic_evaluation/     # Sharpe ratio + economic value (Δ) calculation
│   ├── tests/                   # MCS test, significance test, descriptive stats
│   └── visualization/           # BMA/LASSO heatmaps, prediction plots
├── data/
│   └── processed/               # regression_data.xlsx (23 predictors + lnRV, 726 days)
│                               # rv_rk_results.csv, pca_descriptive_stats.xlsx
│                               # economic_evaluation_data.xlsx, keywords_and_variables.txt
├── results/
│   ├── README.md                # Results documentation
│   ├── RESULTS_SUMMARY.md       # Detailed results with tables (Chinese+English)
│   ├── model_comparison/        # all_models_statistical_metrics.csv (39 rows)
│   ├── mcs_test/                # MCS test results (3 Excel files)
│   ├── economic_evaluation/     # Sharpe ratio + Δ results
│   └── figures/                 # 11 key figures (PCA loadings, predictions, VIP)
├── references/
│   └── reference_paper.pdf      # Core reference: air pollution, weather & investor attention
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE (MIT)
```

## Methods | 研究方法

### 1. Realized Volatility (RV)
- Computed from 5-minute high-frequency returns: RV_t = sqrt(Σ r²)
- Realized Kernel (RK) as robustness check
- Natural log transform: lnRV as dependent variable

### 2. HAR Family Models
Baseline HAR: lnRV_{t+1} = β₀ + β₁lnRV_t + β₂lnRV_{t-4:t} + β₃lnRV_{t-21:t} + ε

Extended models with climate factors:
- **HAR-avgplt**: average air pollution variables
- **HAR-weightplt**: production-weighted air pollution
- **HAR-weather**: weather variables
- **HAR-climfactor**: climate change attention PCA factors
- **HAR-weafactor**: extreme weather attention PCA factors
- **HAR-all**: all 23 predictors combined

### 3. Variable Selection Methods
| Method | Description |
|--------|-------------|
| **PLS-HAR** | Partial Least Squares, extracts latent factors |
| **LASSO-HAR** | L1 regularization with 5-fold time-series CV |
| **BMA-HAR** | Bayesian Model Averaging, BIC-based model weights |
| **Enet-HAR** | Elastic Net, α=0.5 (L1+L2 mixture) |

### 4. Evaluation
- **Statistical**: MSE, MAE, QLIKE, R²LOG (relative loss < 1 = better than baseline)
- **MCS test**: Model Confidence Set, larger p-value = more likely in optimal set
- **Economic**: Sharpe Ratio, economic value (Δ), risk aversion γ=3 and γ=10

## Predictors | 预测因子 (23 total)

| Category | Count | Description |
|----------|-------|-------------|
| Average air pollution | 7 | PM2.5, PM10, AQI, etc. (simple average across production cities) |
| Weighted air pollution | 3 | Production-weighted PM2.5, PM10, AQI |
| Weighted weather | 7 | Temp, humidity, wind, precipitation, cloud, pressure, typhoon dummy |
| Climate change attention | 3 | PCA top-3 components (CLM_PC1~3) |
| Extreme weather attention | 3 | PCA top-3 components (WEA_PC1~3) |

## Getting Started | 快速开始

### Prerequisites | 环境要求
```bash
pip install -r requirements.txt
```

### Data | 数据准备
The main regression dataset is at `data/processed/regression_data.xlsx`, containing all 23 predictors and lnRV for 726 trading days.

主要回归数据位于 `data/processed/regression_data.xlsx`，包含全部23个预测因子和726个交易日的lnRV。

### Running Models | 运行模型
```bash
# HAR baseline and extended models
python code/har_models/har_models.py

# Variable selection models
python code/variable_selection/pls_har.py
python code/variable_selection/lasso_har.py
python code/variable_selection/bma_har.py
python code/variable_selection/enet_har.py
```

## Key Findings | 核心结论

1. HAR models augmented with climate risk factors outperform the baseline HAR across most forecast horizons.
2. BMA and LASSO achieve the best performance, effectively identifying important predictors.
3. Pollution and weather variables contribute significantly in short-horizon forecasts; investor attention variables add incremental power in medium-to-long horizons.
4. Economic value evaluation shows climate-augmented models deliver substantial utility gains for risk-averse investors.

## Citation | 引用

If you use this code or data, please cite:
```
Zhang, H. (2026). Climate Risk and Industrial Silicon Futures Volatility.
Bachelor's Thesis, Central University of Finance and Economics.
```

## License | 许可证

MIT License - see [LICENSE](LICENSE) for details.
