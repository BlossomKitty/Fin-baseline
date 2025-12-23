import pandas as pd

aapl_path = r"D:\FinML\out\features_daily\AAPL.parquet"
panel_path = r"D:\FinML\out\features_daily\_panel.parquet"
cal_path = r"D:\FinML\out\alpaca_calendar.csv"

print("=== AAPL.parquet ===")
df = pd.read_parquet(aapl_path)
print("shape:", df.shape)
print("columns:", len(df.columns))
print("head:")
print(df.head(3))
print("tail:")
print(df.tail(3))

print("\n=== non-null ratio (top 20) ===")
nn = df.notna().mean().sort_values(ascending=False).head(20)
print(nn)

print("\n=== Calendar alignment check ===")
cal = pd.read_csv(cal_path)
cal["timestamp"] = pd.to_datetime(cal["timestamp"])
idx = pd.to_datetime(df.index)
cal_set = set(cal["timestamp"].dt.strftime("%Y-%m-%d"))
idx_set = set(idx.strftime("%Y-%m-%d"))
print("calendar days:", len(cal), cal["timestamp"].min().date(), cal["timestamp"].max().date())
print("AAPL index days:", len(idx), idx.min().date(), idx.max().date())
print("index not in calendar:", len(idx_set - cal_set))

print("\n=== _panel.parquet ===")
panel = pd.read_parquet(panel_path)
print("shape:", panel.shape)
print("unique symbols:", panel["symbol"].nunique())
print("date range:", panel["date"].min(), panel["date"].max())
print("head:")
print(panel.head(3))
