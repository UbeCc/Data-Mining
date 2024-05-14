# public
# 字段名	字段说明
# loan_id	贷款记录唯一标识
# user_id	借款人唯一标识
# total_loan	贷款数额
# year_of_loan	贷款年份
# interest	当前贷款利率
# monthly_payment	分期付款金额
# grade	贷款级别
# employment_type	所在公司类型（世界五百强、国有企业、普通企业…）
# industry	工作领域（传统工业、商业、互联网、金融…）
# work_year	工作年限
# home_exist	是否有房
# censor_status	审核情况
# issue_month	贷款发放的月份
# use	贷款用途类别
# post_code	贷款人申请时邮政编码
# region	地区编码
# debt_loan_ratio	债务收入比
# del_in_18month	借款人过去18个月逾期30天以上的违约事件数
# scoring_low	借款人在贷款评分中所属的下限范围
# scoring_high	借款人在贷款评分中所属的上限范围
# known_outstanding_loan	借款人档案中未结信用额度的数量
# known_dero	贬损公共记录的数量
# pub_dero_bankrup	公开记录清除的数量
# recircle_bal	信贷周转余额合计
# recircle_util	循环额度利用率
# initial_list_status	贷款的初始列表状态
# app_type	是否个人申请
# earlies_credit_mon	借款人最早报告的信用额度开立的月份
# title	借款人提供的贷款名称
# policy_code	公开可用的策略_代码=1新产品不公开可用的策略_代码=2
# f系列匿名特征	匿名特征f0-f4，为一些贷款人行为计数特征的处理
# early_return	借款人提前还款次数
# early_return_amount	贷款人提前还款累积金额
# early_return_amount_3mon	近3个月内提前还款金额
# internet.csv 某网络信用贷产品违约记录数据

# internet
# 字段名	字段说明
# loan_id	网络贷款记录唯一标识
# user_id	用户唯一标识
# total_loan	网络贷款金额
# year_of_loan	网络贷款期限（year）
# interest	网络贷款利率
# monthly_payment	分期付款金额
# class	网络贷款等级
# sub_class	网络贷款等级之子级
# work_type	工作类型（公务员、企业白领、创业…）
# employment_type	所在公司类型（世界五百强、国有企业、普通企业…）
# industry	工作领域（传统工业、商业、互联网、金融…）
# work_year	就业年限（年）
# house_ownership	是否有房
# house_loan_status	房屋贷款状况（无房贷、正在还房贷、已经还完房贷）
# censor_status	验证状态
# marriage	婚姻状态（未婚、已婚、离异、丧偶）
# offsprings	子女状态(无子女、学前、小学、中学、大学、工作)
# issue_date	网络贷款发放的月份
# use	贷款用途
# post_code	借款人邮政编码的前3位
# region	地区编码
# debt_loan_ratio	债务收入比
# del_in_18month	借款人过去18个月信用档案中逾期60天内的违约事件数
# scoring_low	借款人在信用评分系统所属的下限范围
# scoring_high	借款人在信用评分系统所属的上限范围
# pub_dero_bankrup	公开记录清除的数量
# early_return	提前还款次数
# early_return_amount	提前还款累积金额
# early_return_amount_3mon	近3个月内提前还款金额
# recircle_bal	信贷周转余额合计
# recircle_util	循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额
# initial_list_status	网络贷款的初始列表状态
# earlies_credit_line	网络贷款信用额度开立的月份
# title	借款人提供的网络贷款名称
# policy_code	公开策略=1不公开策略=2
# f系列匿名特征	匿名特征f0-f5，为一些网络贷款人行为计数特征的处理


# public 唯一字段
## known_outstanding_loan：借款人档案中未结信用额度的数量
## known_dero：贬损公共记录的数量
## app_type：是否个人申请

# internet 唯一字段
## sub_class：网络贷款等级之子级
## work_type：工作类型
# house_loan_status：房屋贷款状况（无房贷、正在还房贷、已经还完房贷）
## marriage：婚姻状态（未婚、已婚、离异、丧偶）
## offsprings：子女状态(无子女、学前、小学、中学、大学、工作)
## f5：匿名特征f0-f5，为一些网络贷款人行为计数特征的处理，不可合并！

from sklearn.feature_selection import VarianceThreshold
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import shap

def get_test_csv(df, n=1):
    print(df.tail(n))
    df.tail(n).to_csv("test.csv")

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

# internet 唯一字段
## sub_class：网络贷款等级之子级
## work_type：工作类型
# house_loan_status：房屋贷款状况（无房贷、正在还房贷、已经还完房贷）
## marriage：婚姻状态（未婚、已婚、离异、丧偶）
## offsprings：子女状态(无子女、学前、小学、中学、大学、工作)
## f5：匿名特征f0-f5，为一些网络贷款人行为计数特征的处理，不可合并！
def convert_internet(data_path):
    label_encoder = LabelEncoder()
    ohe = OneHotEncoder()
    raw_data = pd.read_csv(data_path)
    data = raw_data.drop(columns=["is_default"])
    data = data.drop(columns=["loan_id", "user_id", "sub_class", "work_type", "house_loan_status", "marriage", "offsprings", "f0", "f1", "f2", "f3", "f4", "f5"])
    # data = data.drop(columns=["earlies_credit_mon"]) # 删掉没用的列
    data["employer_type"] = label_encoder.fit_transform(data["employer_type"])
    data["industry"] = label_encoder.fit_transform(data["industry"])
    data["class"] = label_encoder.fit_transform(data["class"])
    employer_type_oh = ohe.fit_transform(data[["employer_type"]]).toarray()
    industry_oh = ohe.fit_transform(data[["industry"]]).toarray()
    class_oh = ohe.fit_transform(data[["class"]]).toarray() 

    data = pd.concat(
        [
            data,
            pd.DataFrame(employer_type_oh, columns=[f"employer_type_{i}" for i in range(employer_type_oh.shape[1])]),
            pd.DataFrame(industry_oh, columns=[f"industry_{i}" for i in range(industry_oh.shape[1])]),
            pd.DataFrame(class_oh, columns=[f"class_{i}" for i in range(class_oh.shape[1])]),
        ],
        axis=1,
    )

    data["issue_year"] = pd.to_datetime(data["issue_date"], format="%Y-%m-%d").dt.year
    data["issue_month"] = pd.to_datetime(data["issue_date"], format="%Y-%m-%d").dt.month
    data["issue_day"] = pd.to_datetime(data["issue_date"], format="%Y-%m-%d").dt.day
    data["issue_date_days"] = (datetime.datetime.now() - pd.to_datetime(data["issue_date"], format="%Y-%m-%d")).dt.days
    data["total_loan_per_year"] = data["total_loan"] / data["year_of_loan"]
    data["monthly_payment_per_thousand"] = data["monthly_payment"] / (data["total_loan"] / 1000)
    data["work_year"] = data["work_year"].apply(convert_work_year)
    data["work_year"] = data["work_year"].interpolate()
    data = pd.concat([data, pd.DataFrame(data["work_year"], columns=["work_year"])], axis=1)
    data = data.drop("issue_date", axis=1)
    data[["earlies_credit_year", "earlies_credit_month"]] = pd.DataFrame(
        data["earlies_credit_mon"].apply(convert_month).tolist(),
        columns=["earlies_credit_year", "earlies_credit_month"],
    )
    data = data.drop("earlies_credit_mon", axis=1)
    
    # 填补空缺值
    data = data.fillna(data.median())
    
    # 特征选择
    selector = VarianceThreshold(threshold=0.0)
    X = selector.fit_transform(data)
    selected_features = data.columns[selector.get_support()]
    data = data[list(selected_features)]

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    data = pd.DataFrame(X, columns=data.columns)

    # 处理类别不平衡
    sm = SMOTE(random_state=3407)
    X, y = sm.fit_resample(data, raw_data["is_default"])
    data = pd.concat([pd.DataFrame(X), pd.Series(y, name="isDefault")], axis=1)

    data.to_csv(r"./preprocessed_train_internet.csv", index=False)

# public 唯一字段
## known_outstanding_loan：借款人档案中未结信用额度的数量
## known_dero：贬损公共记录的数量
## app_type：是否个人申请
def convert_public(data_path, is_train=True):
    label_encoder = LabelEncoder()   
    ohe = OneHotEncoder()
    raw_data = pd.read_csv(data_path)
    if is_train:
        data = raw_data.drop(columns=["isDefault"])
    else:
        data = raw_data
    data = data.drop(columns=["loan_id", "user_id", "known_outstanding_loan", "known_dero", "app_type", "f0", "f1", "f2", "f3", "f4"])
    # data = data.drop(columns=["earlies_credit_mon"]) # 删掉没用的列
    data["employer_type"] = label_encoder.fit_transform(data["employer_type"])
    data["industry"] = label_encoder.fit_transform(data["industry"])
    data["class"] = label_encoder.fit_transform(data["class"])
    employer_type_oh = ohe.fit_transform(data[["employer_type"]]).toarray()
    industry_oh = ohe.fit_transform(data[["industry"]]).toarray()
    class_oh = ohe.fit_transform(data[["class"]]).toarray() 

    data = pd.concat(
        [
            data,
            pd.DataFrame(employer_type_oh, columns=[f"employer_type_{i}" for i in range(employer_type_oh.shape[1])]),
            pd.DataFrame(industry_oh, columns=[f"industry_{i}" for i in range(industry_oh.shape[1])]),
            pd.DataFrame(class_oh, columns=[f"class_{i}" for i in range(class_oh.shape[1])]),
        ],
        axis=1,
    )

    data["issue_year"] = pd.to_datetime(data["issue_date"], format="%Y/%m/%d").dt.year
    data["issue_month"] = pd.to_datetime(data["issue_date"], format="%Y/%m/%d").dt.month
    data["issue_day"] = pd.to_datetime(data["issue_date"], format="%Y/%m/%d").dt.day
    data["issue_date_days"] = (datetime.datetime.now() - pd.to_datetime(data["issue_date"], format="%Y/%m/%d")).dt.days
    data["total_loan_per_year"] = data["total_loan"] / data["year_of_loan"]
    data["monthly_payment_per_thousand"] = data["monthly_payment"] / (data["total_loan"] / 1000)
    data["work_year"] = data["work_year"].apply(convert_work_year)
    data["work_year"] = data["work_year"].interpolate()
    data = pd.concat([data, pd.DataFrame(data["work_year"], columns=["work_year"])], axis=1)
    data = data.drop("issue_date", axis=1)
    data[["earlies_credit_year", "earlies_credit_month"]] = pd.DataFrame(
        data["earlies_credit_mon"].apply(convert_month).tolist(),
        columns=["earlies_credit_year", "earlies_credit_month"],
    )
    data = data.drop("earlies_credit_mon", axis=1)
    
    # 填补空缺值
    data = data.fillna(data.median())
    
    # 特征选择
    selector = VarianceThreshold(threshold=0.0)
    X = selector.fit_transform(data)
    selected_features = data.columns[selector.get_support()]
    data = data[list(selected_features)]

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    data = pd.DataFrame(X, columns=data.columns)

    # 处理类别不平衡
    if is_train:
        sm = SMOTE(random_state=3407)
        X, y = sm.fit_resample(data, raw_data["isDefault"])
        data = pd.concat([pd.DataFrame(X), pd.Series(y, name="isDefault")], axis=1)
    else:
        data = pd.DataFrame(X, columns=data.columns)

    if is_train:
        data.to_csv(r"./preprocessed_train_public.csv", index=False)
    else:
        data.to_csv(r"./preprocessed_test_public.csv", index=False)

# convert_public("./train_public.csv")
convert_internet("./train_internet.csv")
exit()
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

label_encoder = LabelEncoder()
ohe = OneHotEncoder()
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
