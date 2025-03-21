import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_html("https://www.westmetall.com/en/markdaten.php?action=table&field=LME_Cu_cash")
table = pd.concat(data, ignore_index=True)

table = table[table["date"]!= "date"]
table["date"] = pd.to_datetime(table["date"], format="%d. %B %Y")  
table["year"] = table["date"].dt.year
table["month"] = table["date"].dt.month
table["day"] = table["date"].dt.day
table[["LME Copper Cash-Settlement", "LME Copper 3-month", "LME Copper stock"]] = table[[
    "LME Copper Cash-Settlement", "LME Copper 3-month", "LME Copper stock"
]].apply(pd.to_numeric, errors="coerce")

table = table.dropna()

X = table[["LME Copper stock", "year", "month", "day"]]
y = table["LME Copper Cash-Settlement"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective="reg:squarederror",
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth = 8,
                         subsample = 0.9,
                         colsample_bytree = 0.9)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = r2_score(y_test, y_pred) 
rmse = mse ** 0.5 

table_view = table[["date", "LME Copper Cash-Settlement", "LME Copper 3-month", "LME Copper stock"]]


print(rmse)
