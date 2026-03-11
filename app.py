import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


st.set_page_config(page_title="Diamond Price Prediction", layout="centered")

st.title("💎 Diamond Price Prediction")
st.write("Train-Test Split yang digunakan: **80% Training - 20% Testing**")

file = st.file_uploader("Upload diamonds.csv", type=["csv"])


if file:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    predictor_cols = ['carat','depth','table','x','y','z']

    df_cleaned = df.copy()

    # REMOVE OUTLIERS
    for col in predictor_cols:

        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_cleaned = df_cleaned[
            (df_cleaned[col] >= lower) &
            (df_cleaned[col] <= upper)
        ]


    # ENCODING (SAMA SEPERTI DI FILE ASLI)
    encoder = OrdinalEncoder(
        categories=[
            ['Fair','Good','Very Good','Premium','Ideal'],
            ['J','I','H','G','F','E','D'],
            ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
        ]
    )

    df_cleaned[['cut','color','clarity']] = encoder.fit_transform(
        df_cleaned[['cut','color','clarity']]
    )


    X = df_cleaned.drop('price', axis=1)
    y = df_cleaned['price']


    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    algo = st.selectbox(
        "Pilih Algoritma",
        ["KNN","Random Forest","XGBoost"]
    )


    if st.button("Train Model"):

        with st.spinner("Training model..."):

            if algo == "KNN":

                model = KNeighborsRegressor(
                    n_neighbors=5,
                    weights='distance'
                )

                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)

            elif algo == "Random Forest":

                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            else:

                model = XGBRegressor(
                    learning_rate=0.1,
                    n_estimators=80,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)
                preds = model.predict(X_test)


        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["algo"] = algo
        st.session_state["encoder"] = encoder


        mae = mean_absolute_error(y_test,preds)
        rmse = np.sqrt(mean_squared_error(y_test,preds))
        r2 = r2_score(y_test,preds)


        st.subheader("📊 Model Evaluation")

        col1,col2,col3 = st.columns(3)

        col1.metric("MAE",round(mae,2))
        col2.metric("RMSE",round(rmse,2))
        col3.metric("R²",round(r2,3))


    # PREDICTION
    if "model" in st.session_state:

        st.subheader(f"💎 Predict Diamond Price ({st.session_state['algo']})")

        carat = st.number_input("Carat",0.0,5.0,1.0)
        depth = st.number_input("Depth",40.0,80.0,60.0)
        table = st.number_input("Table",40.0,80.0,55.0)
        x = st.number_input("Length (x)",0.0,10.0,5.0)
        y_val = st.number_input("Width (y)",0.0,10.0,5.0)
        z = st.number_input("Height (z)",0.0,10.0,3.0)

        cut = st.selectbox("Cut",['Fair','Good','Very Good','Premium','Ideal'])
        color = st.selectbox("Color",['J','I','H','G','F','E','D'])
        clarity = st.selectbox("Clarity",['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])


        if st.button("Predict Price"):

            encoder = st.session_state["encoder"]

            input_data = pd.DataFrame({
                "carat":[carat],
                "cut":[cut],
                "color":[color],
                "clarity":[clarity],
                "depth":[depth],
                "table":[table],
                "x":[x],
                "y":[y_val],
                "z":[z]
            })

            input_data[['cut','color','clarity']] = encoder.transform(
                input_data[['cut','color','clarity']]
            )


            if st.session_state["algo"] == "KNN":

                input_scaled = st.session_state["scaler"].transform(input_data)
                prediction = st.session_state["model"].predict(input_scaled)

            else:

                prediction = st.session_state["model"].predict(input_data)


            st.success(f"💰 Predicted Diamond Price: ${prediction[0]:.2f}")
