import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocessing(df):
    df = df.rename(columns={"timestamp": "date"})

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = (
        df[numeric_cols].interpolate().ffill().bfill()
    )  # only interpolate numeric cols

    scaled_features = [
        "trend",
        "workload_intensity",
        "demand_forecast",
        "promotion_intensity",
        "shock_risk",
        "queue_pressure_forecast",
        "network_pressure_forecast",
        "event_load_forecast",
        "event_load_forecast",
        "nominal_capacity",
        "target",
    ]
    scaler = StandardScaler()
    df[scaled_features] = scaler.fit_transform(df[scaled_features])

    le = LabelEncoder()
    df["series_id"] = le.fit_transform(df["series_id"])

    df.to_csv("./dataset/train_processed.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv("dataset/train.csv")
    preprocessing(df)
