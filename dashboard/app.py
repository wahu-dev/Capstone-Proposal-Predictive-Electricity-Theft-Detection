import streamlit as st          # Streamlit library for creating interactive web apps/dashboards
import pandas as pd             # Pandas for data manipulation and loading CSVs
import numpy as np              # NumPy for numerical operations
import pickle                   # Pickle for loading/saving Python objects (e.g., trained models)
import shap                     # SHAP library for explainable AI (feature contributions)
import matplotlib.pyplot as plt # Matplotlib for plotting static charts
import plotly.express as px     # Plotly Express for interactive charts

# Page Configuration
st.set_page_config(
    page_title="Electricity Theft Detection Dashboard",  # Browser tab title
    page_icon="⚡",                                     # Tab icon
    layout="centered",                                  # Layout style of the dashboard
    initial_sidebar_state="expanded"                    # Sidebar default state
)

st.title("⚡ Electricity Theft Detection Dashboard⚡")   # Main dashboard title
st.markdown("Interactive, explainable AI dashboard for detecting electricity theft with actionable insights.") 
# Description under title

# Cached Loaders
@st.cache_data
def load_features():                                    # Loads feature dataset and caches it
    return pd.read_csv("data/processed/final_features.csv")

@st.cache_data
def load_risk_scores():                                 # Loads precomputed risk scores and caches
    return pd.read_csv("data/processed/risk_scores.csv")

@st.cache_resource
def load_model():                                      # Loads trained ensemble model and caches it
    with open("models/ensemble_model.pkl", "rb") as f:
        return pickle.load(f)

# Load Data
features_df = load_features()                           # Features data
risk_df = load_risk_scores()                            # Risk scores
model = load_model()                                    # Ensemble model

dashboard_df = features_df.merge(                        # Combine features with risk scores
    risk_df[["meter_id", "risk_score", "risk_category"]],
    on="meter_id",
    how="left"
)

X = dashboard_df.drop(columns=["meter_id", "is_theft", "risk_score", "risk_category"], errors="ignore")
# Feature matrix for model predictions

# Tabs for clean layout
tab1, tab2, tab3, tab4 = st.tabs([
    "Portfolio Overview ⚡", 
    "Customer Insights 👤", 
    "Explainable Reasoning 🔎", 
    "Policy Simulator 💰"
])

# Three separate sections of the dashboard

# Tab 1: Portfolio Overview
with tab1:
    st.subheader("Portfolio Risk Overview")             # Subheading for overview

    # Metrics: total customers, high-risk count, average risk score
    col1, col2, col3 = st.columns(3)
    total_customers = len(dashboard_df)
    high_risk_count = (dashboard_df['risk_category'] == 'High Risk').sum()
    avg_risk_score = dashboard_df['risk_score'].mean()
    col1.metric("Total Customers", f"{total_customers:,}", "⚡")
    col2.metric("High-Risk Customers", f"{high_risk_count:,}", "⚡")
    col3.metric("Average Risk Score", f"{avg_risk_score:.2f}", "⚡")

    # Histogram of risk scores by category
    st.markdown("### Risk Score Distribution")
    fig = px.histogram(
        dashboard_df,
        x="risk_score",
        color="risk_category",
        nbins=30,
        color_discrete_map={"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"},
        title="Distribution of Electricity Theft Risk Scores"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Global SHAP Feature Importance
    st.markdown("### Global Feature Importance")

    @st.cache_data
    def compute_global_shap(X_sample, _explainer):     # Computes mean absolute SHAP values for features
        shap_vals = _explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_matrix = shap_vals[1]  # class 1
        elif shap_vals.ndim == 3:
            shap_matrix = shap_vals[:, :, 1]
        else:
            shap_matrix = shap_vals
        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": X_sample.columns,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False)
        return importance_df

    X_sample = X.sample(min(100, len(X)), random_state=42) # Sample features for SHAP
    rf_model = model.named_estimators_["rf"].named_steps["model"]  # Extract RF from ensemble
    explainer_global = shap.TreeExplainer(rf_model)      # SHAP explainer
    importance = compute_global_shap(X_sample, explainer_global)

    # Bar chart of top 15 SHAP features
    fig2 = px.bar(
        importance.head(15),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        hover_data={"feature": True, "importance": ":.4f"},
        title="Top 15 Features Driving Theft Predictions"
    )
    st.plotly_chart(fig2, use_container_width=True)

# Tab 2: Customer Insights
with tab2:
    st.subheader("Customer Risk Table & Inspection Simulator")

    # Filter customers by risk category
    risk_filter = st.multiselect(
        "Filter by Risk Category",
        options=dashboard_df["risk_category"].unique(),
        default=list(dashboard_df["risk_category"].unique())
    )
    filtered_df = dashboard_df[dashboard_df["risk_category"].isin(risk_filter)]

    # Conditional coloring for risk categories in table
    def color_risk(val):
        color = "green" if val == "Low Risk" else "orange" if val == "Medium Risk" else "red"
        return f"color: {color}"

    st.dataframe(
        filtered_df[["meter_id", "risk_score", "risk_category"]].style.applymap(color_risk, subset=["risk_category"]),
        height=350
    )

    # Inspection simulator inputs
    st.markdown("### Inspection Strategy Simulator")
    inspection_capacity = st.slider(
        "Number of Inspections Available", min_value=10, max_value=500, value=100, step=10
    )
    strategy = st.radio(
        "Select Inspection Strategy", ["Top Risk", "Random", "Mixed"]
    )

    # Determine which customers to inspect
    if strategy == "Top Risk":
        priority_df = filtered_df.sort_values("risk_score", ascending=False).head(inspection_capacity)
    elif strategy == "Random":
        priority_df = filtered_df.sample(inspection_capacity, random_state=42)
    else:  # Mixed
        top_count = int(inspection_capacity * 0.7)
        random_count = inspection_capacity - top_count
        priority_df = pd.concat([
            filtered_df.sort_values("risk_score", ascending=False).head(top_count),
            filtered_df.sample(random_count, random_state=42)
        ])

    expected_hits = (priority_df["risk_score"] > 0.7).sum()  # Estimate high-risk detections

    st.success(
        f"Recommended inspections: **{inspection_capacity} customers**\n"
        f"Expected high-risk detections: **{expected_hits} customers**"
    )
    st.dataframe(priority_df[["meter_id", "risk_score", "risk_category"]], height=300)

# Tab 3: Explainable AI (Individual Customer)
with tab3:
    st.subheader("Explainable AI – Individual Customer Analysis")

    selected_meter = st.selectbox("Select a Customer (Meter ID)", dashboard_df["meter_id"])
    customer_row = dashboard_df[dashboard_df["meter_id"] == selected_meter]
    X_customer = X.loc[customer_row.index]

    rf_model = model.named_estimators_["rf"].named_steps["model"]
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_customer)

    # Normalize SHAP values for plotting
    if isinstance(shap_values, list):
        shap_vector = shap_values[1][0]  # class 1
        base_value = explainer.expected_value[1]
    elif shap_values.ndim == 3:
        shap_vector = shap_values[0, :, 1]
        base_value = explainer.expected_value[1]
    else:
        shap_vector = shap_values[0]
        base_value = explainer.expected_value

    explanation = shap.Explanation(
        values=shap_vector,
        base_values=base_value,
        data=X_customer.iloc[0],
        feature_names=X_customer.columns
    )

    # Display prediction and SHAP waterfall
    st.markdown(
        f"**Meter ID:** `{selected_meter}`  \n"
        f"**Predicted Risk Score:** `{customer_row['risk_score'].values[0]:.2f}`  \n"
        f"**Risk Category:** `{customer_row['risk_category'].values[0]}`"
    )

    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig, bbox_inches="tight")

# Tab 4: Cost–Benefit Policy Simulator
@st.cache_data
def compute_impact_curve(
    y_true,
    y_proba,
    n_customers,
    theft_rate,
    annual_bill,
    inspection_cost,
    max_inspections
):
    thresholds = np.linspace(0, 1, 101)
    n_theft_customers = n_customers * theft_rate
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        recall = tp / max(1, np.sum(y_true == 1))
        precision = tp / max(1, tp + fp)

        suspected = n_theft_customers * recall
        inspected = min(suspected, max_inspections)

        gross = inspected * annual_bill
        cost = inspected * inspection_cost
        net = gross - cost
        roi = net / cost if cost > 0 else 0

        results.append({
            "threshold": t,
            "recall": recall,
            "precision": precision,
            "inspected": inspected,
            "gross_B": gross / 1e9,
            "cost_M": cost / 1e6,
            "net_B": net / 1e9,
            "roi": roi
        })

    return pd.DataFrame(results)


with tab4:
    st.subheader("💰 Cost–Benefit & Inspection Policy Simulator")

    # -------------------------
    # Policy Inputs
    # -------------------------
    c1, c2, c3 = st.columns(3)

    inspection_cap = c1.slider(
        "Annual Inspection Capacity",
        min_value=50_000,
        max_value=800_000,
        value=500_000,
        step=50_000
    )

    inspection_cost = c2.number_input(
        "Cost per Inspection (KES)",
        min_value=100,
        max_value=2_000,
        value=500,
        step=50
    )

    avg_monthly_bill = c3.number_input(
        "Average Monthly Bill (KES)",
        min_value=1_000,
        max_value=10_000,
        value=3_000,
        step=500
    )

    # -------------------------
    # Fixed Assumptions
    # -------------------------
    n_customers = 10_000_000
    theft_rate = 0.05
    annual_bill = avg_monthly_bill * 12

    # -------------------------
    # Simulated Inspections Summary
    # -------------------------
    n_theft_customers = n_customers * theft_rate
    inspected_customers = min(inspection_cap, int(n_theft_customers))
    gross_recovery = inspected_customers * annual_bill
    total_cost = inspected_customers * inspection_cost
    net_impact = gross_recovery - total_cost
    roi = net_impact / total_cost if total_cost > 0 else 0

    # -------------------------
    # Display Metrics
    # -------------------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers Inspected", f"{inspected_customers:,}")
    k2.metric("Gross Recovery(B)", f"{gross_recovery/1e9:.2f}")
    k3.metric("Inspection Cost(M)", f"{total_cost/1e6:.1f}")  # <- Added total cost
    k4.metric("Net Impact (KES B)", f"{net_impact/1e9:.2f}")
    k5.metric("ROI (x)", f"{roi:.1f}")

    # -------------------------
    # ROI Formula
    # -------------------------
    st.markdown(
        """
**ROI Formula (Display):**  

$$
ROI = \\frac{\\text{Net Impact}}{\\text{Total Cost}} = \\frac{\\text{Gross Recovery} - \\text{Total Cost}}{\\text{Total Cost}}
$$

Where:  
- **Gross Recovery** = Inspected Customers × Annual Bill  
- **Total Cost** = Inspected Customers × Inspection Cost  
- **Net Impact** = Gross Recovery − Total Cost
""",
    unsafe_allow_html=True
)

    # -------------------------
    # Executive Summary Text
    # -------------------------
    st.success(
        f"""### 📌 Executive Summary

- **Estimated Inspections:** {inspected_customers:,} per year
- **Gross Recovery:** KES {gross_recovery/1e9:.2f}B
- **Inspection Cost:** KES {total_cost/1e6:.1f}M
- **Net Impact:** KES {net_impact/1e9:.2f}B
- **ROI:** {roi:.1f}×

This provides a simplified, **policy simulator**."""
    )

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(
    "Interactive dashboard for electricity theft detection. All metrics are read-only "
    "and derived from precomputed models and SHAP explainability."
)
