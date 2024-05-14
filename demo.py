import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def convert_work_year(x):
    if isinstance(x, str) and x.strip():
        if x == "< 1 year":
            return 0
        elif x == "1 year":
            return 1
        elif x == "10+ years":
            return 10
        elif x.endswith(" years"):
            return int(x.split(" ")[0])
        else:
            return None
    else:
        return None

def convert_month(x):
    if isinstance(x, str) and "-" in x:
        parts = x.split("-")
        if len(parts) == 2:
            if parts[0].isdigit():
                return (
                    2000 + int(parts[0]),
                    {
                        "Jan": 1,
                        "Feb": 2,
                        "Mar": 3,
                        "Apr": 4,
                        "May": 5,
                        "Jun": 6,
                        "Jul": 7,
                        "Aug": 8,
                        "Sep": 9,
                        "Oct": 10,
                        "Nov": 11,
                        "Dec": 12,
                    }[parts[1]],
                )
            else:
                if parts[1] == "00":
                    return (
                        2000,
                        {
                            "Jan": 1,
                            "Feb": 2,
                            "Mar": 3,
                            "Apr": 4,
                            "May": 5,
                            "Jun": 6,
                            "Jul": 7,
                            "Aug": 8,
                            "Sep": 9,
                            "Oct": 10,
                            "Nov": 11,
                            "Dec": 12,
                        }[parts[0]],
                    )
                else:
                    return (
                        1900 + int(parts[1]),
                        {
                            "Jan": 1,
                            "Feb": 2,
                            "Mar": 3,
                            "Apr": 4,
                            "May": 5,
                            "Jun": 6,
                            "Jul": 7,
                            "Aug": 8,
                            "Sep": 9,
                            "Oct": 10,
                            "Nov": 11,
                            "Dec": 12,
                        }[parts[0]],
                    )
        else:
            return None, None
    else:
        return None, None

test_data = pd.read_csv(r"./test_public.csv")
label_encoder = LabelEncoder()
test_data["employer_type"] = label_encoder.fit_transform(test_data["employer_type"])
test_data["industry"] = label_encoder.fit_transform(test_data["industry"])
test_data["class"] = label_encoder.fit_transform(test_data["class"])

ohe = OneHotEncoder()
test_employer_type_oh = ohe.fit_transform(test_data[["employer_type"]]).toarray()
test_industry_oh = ohe.fit_transform(test_data[["industry"]]).toarray()
test_class_oh = ohe.fit_transform(test_data[["class"]]).toarray()

test_data = pd.concat(
    [
        test_data,
        pd.DataFrame(
            test_employer_type_oh,
            columns=[
                f"employer_type_{i}" for i in range(test_employer_type_oh.shape[1])
            ],
        ),
    ],
    axis=1,
)
test_data = pd.concat(
    [
        test_data,
        pd.DataFrame(
            test_industry_oh,
            columns=[f"industry_{i}" for i in range(test_industry_oh.shape[1])],
        ),
    ],
    axis=1,
)
test_data = pd.concat(
    [
        test_data,
        pd.DataFrame(
            test_class_oh,
            columns=[f"class_{i}" for i in range(test_class_oh.shape[1])],
        ),
    ],
    axis=1,
)

# # 处理日期特征
import datetime

test_data["issue_year"] = pd.to_datetime(
    test_data["issue_date"], format="%Y/%m/%d"
).dt.year
test_data["issue_month"] = pd.to_datetime(
    test_data["issue_date"], format="%Y/%m/%d"
).dt.month
test_data["issue_day"] = pd.to_datetime(
    test_data["issue_date"], format="%Y/%m/%d"
).dt.day
test_data["issue_date_days"] = (
    datetime.datetime.now()
    - pd.to_datetime(test_data["issue_date"], format="%Y/%m/%d")
).dt.days

# 特征工程
test_data["total_loan_per_year"] = (
    test_data["total_loan"] / test_data["year_of_loan"]
)
test_data["monthly_payment_per_thousand"] = test_data["monthly_payment"] / (
    test_data["total_loan"] / 1000
)

test_data["work_year"] = test_data["work_year"].apply(convert_work_year)
test_data["work_year"] = test_data["work_year"].interpolate()
test_data = pd.concat(
    [test_data, pd.DataFrame(test_data["work_year"], columns=["work_year"])], axis=1
)
test_data = test_data.drop("issue_date", axis=1)

test_data[["earlies_credit_year", "earlies_credit_month"]] = pd.DataFrame(
    test_data["earlies_credit_mon"].apply(convert_month).tolist(),
    columns=["earlies_credit_year", "earlies_credit_month"],
)
test_data = test_data.drop("earlies_credit_mon", axis=1)

test_data = test_data.fillna(test_data.median())

# 特征选择
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)

# 数据标准化
scaler = StandardScaler()
test_X = scaler.fit_transform(test_data)
test_data = pd.concat(
    [
        pd.DataFrame(test_X, columns=test_data.columns),
    ],
    axis=1,
)

# 保存预处理后的数据集
test_data.to_csv(r"./preprocessed_test_public.csv", index=False)