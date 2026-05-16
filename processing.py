import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


def preprocess_data(df, scale_mode=None):
    df = df.copy()
    df.drop("customerID", axis=1, inplace=True)
    target = "Churn"
    target_map = {"No": 0, "Yes": 1}
    df[target] = df[target].map(target_map).astype(int)
    df.drop("TotalCharges", axis=1, inplace=True)

    num_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "gender",
    ]

    multi_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaymentMethod",
    ]

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    }

    df = encode_data(
        df,
        binary_cols,
        multi_cols,
        contract_map,
    )

    if scale_mode == "standard":
        df = scale_data(df, StandardScaler, num_cols)

    elif scale_mode == "minmax":
        df = scale_data(df, MinMaxScaler, num_cols)

    return df


def scale_data(df, scaler_cls=StandardScaler, num_cols=None):
    df_scaled = df.copy()
    if not num_cols:
        return df_scaled
    scaler = scaler_cls()
    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    return df_scaled


def encode_data(df, binary_cols=None, multi_cols=None, contract_map=None):
    df = df.copy()
    if binary_cols:
        le = LabelEncoder()
        for col in binary_cols:
            df[col] = le.fit_transform(df[col])
    if multi_cols:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(df[multi_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(multi_cols),
            index=df.index,
        )
        df = df.drop(columns=multi_cols)
        df = pd.concat([df, encoded_df], axis=1)
    if contract_map:
        df["Contract"] = df["Contract"].map(contract_map)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df
