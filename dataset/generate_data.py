import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

store_ids = [1, 2, 3]
product_ids = [101, 102, 103, 104]

rows = []
for store in store_ids:
    for product in product_ids:
        base_sales = np.random.randint(100, 300)
        for date in dates:
            promotion = np.random.choice([0, 1], p=[0.8, 0.2])
            holiday = 1 if date.weekday() == 6 or date.month == 12 and date.day == 25 else 0
            seasonal = 20 * np.sin(2 * np.pi * date.dayofyear / 365)
            weekend_boost = 30 if date.weekday() >= 5 else 0
            promo_boost = 50 * promotion
            noise = np.random.normal(0, 15)
            sales = max(0, int(base_sales + seasonal + weekend_boost + promo_boost + noise))
            rows.append([date.strftime('%Y-%m-%d'), store, product, sales, promotion, holiday])

df = pd.DataFrame(rows, columns=['date', 'store_id', 'product_id', 'sales', 'promotion', 'holiday'])
df.to_csv('/home/claude/retail_sales_forecasting/dataset/sales_data.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df.head())
