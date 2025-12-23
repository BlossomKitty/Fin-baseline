

# FinML Data Pipeline (Alpaca + FMP + DefiLlama) & Baselines

本仓库/项目用于：
1) 下载并整理 **美股日线复权 OHLCV**（Alpaca，516 标的）
2) 拉取 **FMP 基本面与新闻**（含 ratios/key-metrics 的 quarter/annual 历史序列）
3) 将 JSONL 基本面/新闻对齐到 **真实交易日历**，生成日频特征面板（parquet）
4) 在不使用任何 LLM API Key 的前提下跑通 **baseline 回测**（验证框架正确性）
5)（可选补充）拉取 **DeFiLlama 链 TVL** 作为“数字货币/链上基本面”补充数据



## 0. 环境准备（PowerShell）

建议使用 conda 环境：
```powershell
conda activate finml
````

---

## 1) Alpaca：下载近 5 年复权后（日频）OHLCV（含交易日历）

说明：

* `adjustment=all`：考虑分红/拆股等公司行为后的调整口径
* `timeframe=1Day`
* 输出：合并 CSV +（可选）按标的拆分 CSV + 交易日历 CSV

### 1.1 设置环境变量（PowerShell）

```powershell
$env:ALPACA_API_KEY_ID="xxx"
$env:ALPACA_API_SECRET_KEY="yyy"
$env:ALPACA_DATA_FEED="iex"   # 默认 iex；若有 SIP 权限可改 sip
```

### 1.2 批量下载 516 标的 bars（推荐）

```powershell
python D:\FinML\scripts\download_alpaca_bars_bulk.py `
  --symbols-file "D:\FinML\symbols_2025-12-20.txt" `
  --out-dir "D:\FinML\out\alpaca_bars_516" `
  --start 2021-01-01 `
  --chunk-size 200 `
  --adjustment all `
  --per-symbol `
  --make-calendar
```

产出（示例）：

* `D:\FinML\out\alpaca_bars_516\combined_daily_adjusted.csv`
* `D:\FinML\out\alpaca_bars_516\alpaca_calendar.csv`

> 交易日历将用于后续基本面对齐，避免用 `pandas bday` 造成错位。

---

## 2) FMP：拉取基本面 + 新闻（JSONL）

### 2.1 安装依赖

```powershell
pip install -r requirements.txt
```

### 2.2 设置 FMP API Key

```powershell
$env:FMP_API_KEY="YOUR_KEY"
```

### 2.3 运行：拉 fundamentals（含历史序列 + 支持 overwrite/news）

你现在的目标类型（历史序列，非 TTM）应至少包含：

* `ratios_quarter`, `ratios_annual`
* `key_metrics_quarter`, `key_metrics_annual`
  并支持参数：
* `--fetch-news`
* `--overwrite`
  且使用 **stable 端点**（不是 `/api/v3/...`）

#### 2.3.1 只拉财报/预测/filings（不含新闻）

```powershell
python scripts/download_fmp_fundamentals.py `
  --symbols-file symbols_2025-12-20.txt `
  --out-dir out/fmp_jsonl `
  --start 2021-01-01 `
  --workers 6
```

#### 2.3.2 拉 fundamentals + press releases + stock news（推荐你现在使用的 out）

```powershell
python scripts/download_fmp_fundamentals.py `
  --symbols-file symbols_2025-12-20.txt `
  --out-dir out/fmp_jsonl_news `
  --start 2021-01-01 `
  --workers 6 `
  --fetch-news
```

#### 2.3.3 如需强制重拉（覆盖旧结果）

```powershell
python scripts/download_fmp_fundamentals.py `
  --symbols-file symbols_2025-12-20.txt `
  --out-dir out/fmp_jsonl_news `
  --start 2021-01-01 `
  --workers 6 `
  --fetch-news `
  --overwrite
```

---

## 3) 从已有 out（不重跑下载）生成日频特征（features_daily）

你已生成：

* FMP JSONL：`D:\FinML\out\fmp_jsonl_news`
* Alpaca 交易日历：`D:\FinML\out\alpaca_bars_516\alpaca_calendar.csv`

### 3.1 构建日频特征（并输出 combined panel）

```powershell
python scripts/build_features_daily.py `
  --jsonl-dir "D:\FinML\out\fmp_jsonl_news" `
  --out-dir "D:\FinML\out\features_daily" `
  --config "configs\features_v1.json" `
  --start 2021-01-01 `
  --calendar-bars-csv "D:\FinML\out\alpaca_bars_516\alpaca_calendar.csv" `
  --combine
```

产出：

* `D:\FinML\out\features_daily\{SYMBOL}.parquet`（每股日频特征）
* `D:\FinML\out\features_daily\_panel.parquet`（全量长表 panel）

### 3.2 快速检查输出是否正常

```powershell
python D:\FinML\scripts\inspect_features.py
```

检查点：

* `Calendar alignment check`：`index not in calendar` 应为 0
* 特征 NaN：早期 NaN 常见（披露滞后 + 需要足够历史才 ffill），属于预期现象

---

## 4) FMP：拉最近 90 天宏观新闻（可选补充）

用于做宏观/跨市场情绪或风险因子（不依赖公司粒度）。

```powershell
python D:\FinML\scripts\download_fmp_macro_news.py `
  --out-dir "D:\FinML\out\fmp_macro_news_3m" `
  --days 90 `
  --limit 200 `
  --max-pages 200 `
  --types "general,forex,crypto,fmp_articles" `
  --sleep 0.15
```

---

## 5) Baseline 回测（不需要任何 LLM / 4o key）

目的：验证你的回测框架、收益计算、重平衡、交易成本是否正确，并提供可复现实验下限。

### 5.1 Buy & Hold（底线）

```powershell
python D:\FinML\scripts\backtest_baselines.py `
  --ohlcv-csv "D:\FinML\out\alpaca_bars_516\combined_daily_adjusted.csv" `
  --strategy buyhold `
  --start 2021-01-01 `
  --out-dir "D:\FinML\out\bt_results"
```

### 5.2 等权（周频重平衡）

```powershell
python D:\FinML\scripts\backtest_baselines.py `
  --ohlcv-csv "D:\FinML\out\alpaca_bars_516\combined_daily_adjusted.csv" `
  --strategy equal_weight `
  --rebalance weekly `
  --cost-bps 2 `
  --start 2021-01-01 `
  --out-dir "D:\FinML\out\bt_results"
```

### 5.3 动量 top-k（只用 OHLCV）

```powershell
python D:\FinML\scripts\backtest_baselines.py `
  --ohlcv-csv "D:\FinML\out\alpaca_bars_516\combined_daily_adjusted.csv" `
  --strategy mom_topk `
  --lookback 60 `
  --top-k 50 `
  --rebalance weekly `
  --cost-bps 5 `
  --ret-clip 0.2 `
  --min-history 800 `
  --start 2021-01-01 `
  --out-dir "D:\FinML\out\bt_results"
```

### 5.4 基本面 top-k（ratios + key_metrics）

```powershell
python D:\FinML\scripts\backtest_baselines.py `
  --ohlcv-csv "D:\FinML\out\alpaca_bars_516\combined_daily_adjusted.csv" `
  --panel-parquet "D:\FinML\out\features_daily\_panel.parquet" `
  --strategy fund_topk `
  --top-k 50 `
  --rebalance monthly `
  --cost-bps 5 `
  --start 2021-01-01 `
  --out-dir "D:\FinML\out\bt_results"
```

### 5.5 新闻/公告计数 top-k（不做LLM）

```powershell
python D:\FinML\scripts\backtest_baselines.py `
  --ohlcv-csv "D:\FinML\out\alpaca_bars_516\combined_daily_adjusted.csv" `
  --panel-parquet "D:\FinML\out\features_daily\_panel.parquet" `
  --strategy news_topk `
  --top-k 50 `
  --rebalance weekly `
  --cost-bps 10 `
  --start 2021-01-01 `
  --out-dir "D:\FinML\out\bt_results"
```

输出：

* `D:\FinML\out\bt_results\equity_*.csv`

---

## 6) DefiLlama：链 TVL（开放端点，免费限速友好，可选）

定位：补充“数字货币/链上基本面”数据源，尤其适用于你们多智能体框架中：

* 作为风险/流动性 regime 变量（TVL 上升/下降）
* 作为 crypto 市场景气度 proxy（链层资金沉淀）

### 6.1 列出可用链（复制 name 列最稳）

```powershell
python scripts\download_llama_fundamentals.py --out-dir "D:\FinML\out\llama" --list-chains --max-list-rows 50
```

### 6.2 拉指定链 TVL（示例：2021-01-01~2025-12-22）

```powershell
python scripts\download_llama_fundamentals.py `
  --out-dir "D:\FinML\out\llama" `
  --start 2021-01-01 `
  --end 2025-12-22 `
  --chains "Ethereum,Arbitrum,OP Mainnet,Base,Polygon,Avalanche,BSC,Gnosis,Linea,Blast,Mantle,Fraxtal,Taiko,Rootstock,Fantom" `
  --sleep 1.0 `
  --overwrite
```

产出：

* `D:\FinML\out\llama\CHAIN_ETHEREUM.jsonl`
* `D:\FinML\out\llama\CHAIN_ARBITRUM.jsonl`
* ...

---

## 7) 典型数据目录（你当前实际使用）

* OHLCV（Alpaca 516）
  `D:\FinML\out\alpaca_bars_516\combined_daily_adjusted.csv`
  `D:\FinML\out\alpaca_bars_516\alpaca_calendar.csv`

* FMP 公司粒度 JSONL（含 news / ratios / key_metrics）
  `D:\FinML\out\fmp_jsonl_news\*.jsonl`

* 日频特征面板（与交易日历对齐）
  `D:\FinML\out\features_daily\_panel.parquet`

* Baseline 回测结果
  `D:\FinML\out\bt_results\equity_*.csv`

* 宏观新闻（90 天）
  `D:\FinML\out\fmp_macro_news_3m\macro_news_*.jsonl`

* DeFiLlama 链 TVL（可选）
  `D:\FinML\out\llama\CHAIN_*.jsonl`

---

## 8) 常见问题

### Q1：特征里大量 NaN 正常吗？

正常。原因：

* ratios/key_metrics 是季/年频披露，有滞后（lag）；
* 需要累积足够历史后才会 ffill 到日频；
* 回测前期会出现空值是常态。

### Q2：为什么必须用 Alpaca 真实交易日历？

因为不同市场/节假日/停牌会导致 `pandas business day` 与真实交易日不一致，进而造成“特征对齐错误”和未来函数风险。你已通过 `inspect_features.py` 验证 `index not in calendar = 0`。

### Q3：不想重跑 FMP out，只想做后续步骤？

可以。只要 out 目录里已经包含目标 type（ratios_quarter / key_metrics_quarter 等），直接从 `build_features_daily.py` 开始即可。

---

## License / Notes

* 数据来自第三方 API（Alpaca / FMP / DefiLlama）。请遵守各自服务条款与限速策略。
* 本 README 只描述数据与回测管线，不构成投资建议。

```

如果你愿意，我也可以按你当前仓库的**真实文件名**再做一次“对齐版”（比如把 `scripts/` 里实际存在的脚本名、参数名逐个核对），避免 README 和代码参数未来再飘。
::contentReference[oaicite:0]{index=0}
```
