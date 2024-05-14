import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import shap

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

# train_data = pd.read_csv(r"./test_public.csv")
label_encoder = LabelEncoder()
# train_data["employer_type"] = label_encoder.fit_transform(train_data["employer_type"])
# train_data["industry"] = label_encoder.fit_transform(train_data["industry"])
# train_data["class"] = label_encoder.fit_transform(train_data["class"])

ohe = OneHotEncoder()
# train_employer_type_oh = ohe.fit_transform(train_data[["employer_type"]]).toarray()
# train_industry_oh = ohe.fit_transform(train_data[["industry"]]).toarray()
# train_class_oh = ohe.fit_transform(train_data[["class"]]).toarray()

# train_data = pd.concat(
#     [
#         train_data,
#         pd.DataFrame(
#             train_employer_type_oh,
#             columns=[
#                 f"employer_type_{i}" for i in range(train_employer_type_oh.shape[1])
#             ],
#         ),
#     ],
#     axis=1,
# )
# train_data = pd.concat(
#     [
#         train_data,
#         pd.DataFrame(
#             train_industry_oh,
#             columns=[f"industry_{i}" for i in range(train_industry_oh.shape[1])],
#         ),
#     ],
#     axis=1,
# )
# train_data = pd.concat(
#     [
#         train_data,
#         pd.DataFrame(
#             train_class_oh,
#             columns=[f"class_{i}" for i in range(train_class_oh.shape[1])],
#         ),
#     ],
#     axis=1,
# )

# # 处理日期特征
import datetime

# train_data["issue_year"] = pd.to_datetime(
#     train_data["issue_date"], format="%Y/%m/%d"
# ).dt.year
# train_data["issue_month"] = pd.to_datetime(
#     train_data["issue_date"], format="%Y/%m/%d"
# ).dt.month
# train_data["issue_day"] = pd.to_datetime(
#     train_data["issue_date"], format="%Y/%m/%d"
# ).dt.day
# train_data["issue_date_days"] = (
#     datetime.datetime.now()
#     - pd.to_datetime(train_data["issue_date"], format="%Y/%m/%d")
# ).dt.days

# # 特征工程
# train_data["total_loan_per_year"] = (
#     train_data["total_loan"] / train_data["year_of_loan"]
# )
# train_data["monthly_payment_per_thousand"] = train_data["monthly_payment"] / (
#     train_data["total_loan"] / 1000
# )

# train_data["work_year"] = train_data["work_year"].apply(convert_work_year)
# train_data["work_year"] = train_data["work_year"].interpolate()
# train_data = pd.concat(
#     [train_data, pd.DataFrame(train_data["work_year"], columns=["work_year"])], axis=1
# )
# train_data = train_data.drop("issue_date", axis=1)

# train_data[["earlies_credit_year", "earlies_credit_month"]] = pd.DataFrame(
#     train_data["earlies_credit_mon"].apply(convert_month).tolist(),
#     columns=["earlies_credit_year", "earlies_credit_month"],
# )
# train_data = train_data.drop("earlies_credit_mon", axis=1)

# train_data = train_data.fillna(train_data.median())

# # 特征选择
from sklearn.feature_selection import VarianceThreshold

# selector = VarianceThreshold(threshold=0.0)
# train_X = selector.fit_transform(train_data.drop("isDefault", axis=1))
# selected_features = train_data.drop("isDefault", axis=1).columns[selector.get_support()]
# train_data = train_data[["isDefault"] + list(selected_features)]

# # 数据标准化
# scaler = StandardScaler()
# train_X = scaler.fit_transform(train_data.drop("isDefault", axis=1))
# train_data = pd.concat(
#     [
#         pd.DataFrame(train_X, columns=train_data.drop("isDefault", axis=1).columns),
#         train_data["isDefault"],
#     ],
#     axis=1,
# )

# # 处理类别不平衡
# sm = SMOTE(random_state=42)
# train_X, train_y = sm.fit_resample(
#     train_data.drop("isDefault", axis=1), train_data["isDefault"]
# )
# train_data = pd.concat(
#     [pd.DataFrame(train_X), pd.Series(train_y, name="isDefault")], axis=1
# )

# # 划分训练集和测试集
# X = train_data.drop("isDefault", axis=1)
# y = train_data["isDefault"]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 保存预处理后的数据集
# train_data.to_csv(r"./preprocessed_train_public.csv", index=False)

# TODO: 构建 RNN
# 构建模型并训练
model = LogisticRegression()
# model.fit(X_train, y_train)

# # 评估模型
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-score: {f1:.2f}")

# 特征重要性分析
# explainer = shap.LinearExplainer(model, X_train)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar")

# 读取测试数据集
test_data = pd.read_csv(r"./test_public.csv")

# 编码分类特征
test_data["employer_type"] = label_encoder.fit_transform(test_data["employer_type"])
test_data["industry"] = label_encoder.fit_transform(test_data["industry"])
test_data["class"] = label_encoder.fit_transform(test_data["class"])


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
            test_class_oh, columns=[f"class_{i}" for i in range(test_class_oh.shape[1])]
        ),
    ],
    axis=1,
)

# 处理日期特征
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
    datetime.datetime.now() - pd.to_datetime(test_data["issue_date"], format="%Y/%m/%d")
).dt.days
test_data = test_data.drop("issue_date", axis=1)

# 特征工程
test_data["total_loan_per_year"] = test_data["total_loan"] / test_data["year_of_loan"]
test_data["monthly_payment_per_thousand"] = test_data["monthly_payment"] / (
    test_data["total_loan"] / 1000
)

test_data["work_year"] = test_data["work_year"].apply(convert_work_year)
test_data["work_year"] = test_data["work_year"].interpolate()
test_data = pd.concat(
    [test_data, pd.DataFrame(test_data["work_year"], columns=["work_year"])], axis=1
)

test_data[["earlies_credit_year", "earlies_credit_month"]] = pd.DataFrame(
    test_data["earlies_credit_mon"].apply(convert_month).tolist(),
    columns=["earlies_credit_year", "earlies_credit_month"],
)
test_data = test_data.drop("earlies_credit_mon", axis=1)
test_data = test_data.fillna(test_data.median())

test_data.to_csv("preprocessed_test_public.csv", index=False)
# # 特征选择
# test_X = test_data[selected_features]

# # 数据标准化
# test_X = scaler.transform(test_X)

# # 获取预测类别标签
# test_y_pred_labels = model.predict(test_X)

# # 获取预测概率
# test_y_pred_proba = model.predict_proba(test_X)

# print("预测类别标签:")
# print(test_y_pred_labels)

# print("预测概率:")
# print(test_y_pred_proba)

# num_samples = test_y_pred_proba.shape[0]

# # 创建 id 列
# id_col = list(test_data["loan_id"])

# # 创建 DataFrame
# result_df = pd.DataFrame(
#     {"id": id_col, "isDefault": test_y_pred_proba[:, 1]}  # 取第二列作为预测概率
# )

# # 保存为 CSV 文件
# result_df.to_csv(r"./submission.csv", index=False)
