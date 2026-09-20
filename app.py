import streamlit as st
import joblib
import json
import pandas as pd
import shap
import matplotlib.pyplot as plt
from textblob import TextBlob

with open('models/xgb.pkl', 'rb') as f:
    model = joblib.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)
with open('models/feature_list.json', 'r') as f:
    feature_list = json.load(f)
with open('models/agent_freq.pkl', 'rb') as f:
    agent_freq = joblib.load(f)

st.set_page_config(page_title="CSAT Predictor", page_icon="📊", layout="wide")

st.title("📊 CSAT Score Prediction App")
st.caption("Predicts customer satisfaction score based on operational & text features.")

st.markdown("---")

# Layout: input left | results right
left, right = st.columns([1,1.5])

with left:
    st.subheader("🛠️ Enter Scenario")
    customer_remarks = st.text_area("✏️ Customer Remarks", " ")
    response_delay = st.number_input("⏱️ Response Delay (minutes)", 0, 2880)
    order_dayofweek = st.selectbox("📅 Day of Week", range(7))
    order_month = st.selectbox("📆 Order Month", range(1,13))
    order_hour = st.number_input("🕰️ Order Hour", 0, 23)
    agent_name = st.selectbox("👤 Agent", list(agent_freq.keys()))
    is_weekend = st.radio("🏖️ Weekend?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")

    # Auto features
    remark_length = len(customer_remarks.split())
    num_tokens = len(customer_remarks.split())
    sentiment = TextBlob(customer_remarks).sentiment.polarity
    agent_encoded = agent_freq.get(agent_name, 0)

    input_df = pd.DataFrame([[ 
        response_delay, order_dayofweek, order_month, order_hour,
        agent_encoded, is_weekend, remark_length, num_tokens, sentiment
    ]], columns=feature_list)

    # Scale
    features_to_scale = ['num_tokens', 'response_delay_minutes', 'agent_name_encoded', 'remark_length', 'sentiment']
    input_df_scaled = input_df.copy()
    input_df_scaled[features_to_scale] = scaler.transform(input_df[features_to_scale])

    predict_btn = st.button("🚀 Predict")

with right:
    if predict_btn:
        pred = model.predict(input_df_scaled)[0]
        proba = model.predict_proba(input_df_scaled)[0]

        # Sentiment label
        sentiment_label = "Positive 😀" if sentiment > 0.2 else "Negative 😞" if sentiment < -0.2 else "Neutral 😐"

        # Recommendations
        recommendations = []
        if response_delay > 60:
            recommendations.append("⏱️ Reduce response delay to boost CSAT.")
        if sentiment < -0.2:
            recommendations.append("🙏 Apologize and offer compensation.")
        if remark_length < 5:
            recommendations.append("📞 Ask follow-up questions to understand better.")
        if not recommendations:
            recommendations.append("✅ All looks good!")

        st.subheader("📊 Results")
        col1, col2, col3 = st.columns(3)
        col1.success(f"🎯 Predicted CSAT: {pred}")
        col2.warning(f"😟 Dissatisfied (≤2): {proba[1]*100:.1f}%")
        col3.info(f"😊 Satisfied (>2): {proba[0]*100:.1f}%")

        st.markdown("---")

        st.subheader("📝 Sentiment & Tag")
        st.write(f"Text Sentiment Score: **{sentiment:.2f}** → *{sentiment_label}*")

        st.subheader("💡 Recommendations")
        for rec in recommendations:
            st.write(f"- {rec}")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["🌟 Feature Importances", "🧠 SHAP Explanation", "📈 Input Details"])

        with tab1:
            importances = model.feature_importances_
            df = pd.DataFrame({'Feature': feature_list, 'Importance': importances}).sort_values(by='Importance')
            fig, ax = plt.subplots()
            ax.barh(df['Feature'], df['Importance'], color='cornflowerblue')
            ax.set_xlabel("Importance")
            st.pyplot(fig)

        with tab2:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df_scaled)
            shap.initjs()
            fig2 = shap.plots.force(
                explainer.expected_value, shap_values[0], input_df_scaled.iloc[0],
                matplotlib=True, show=False
            )
            st.pyplot(fig2)

        with tab3:
            st.json(input_df.to_dict(orient='records')[0])

    else:
        st.info("⬅️ Fill the inputs and click **Predict** to see results, insights, and recommendations.")
