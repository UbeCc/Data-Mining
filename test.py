# 删掉 work_year.1, loan_id, user_id
import pandas as pd

train_data = pd.read_csv('train_public.csv')
print(train_data["app_type"])
cnt = 0
for i in range(len(train_data.columns)):
    cnt += 1
print(cnt)

train_data = pd.read_csv('train_internet.csv')
print(train_data["work_type"])
cnt = 0
for i in range(len(train_data.columns)):
    cnt += 1
print(cnt)