import streamlit as st
import pandas as pd

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

st.title("🧭 Epidemiology Decision Simulator")
st.markdown("Think structurally: Outcome → Time → Design → Predictor → Measure → Interpretation")

# ==========================================================
# STEP 1: OUTCOME
# ==========================================================

st.subheader("Step 1️⃣: Identify the Outcome")

outcome_type = st.selectbox(
    "What type of outcome variable?",
    ["Binary", "Continuous", "Rate (person-time)"]
)

# ==========================================================
# STEP 2: TEMPORALITY
# ==========================================================

st.subheader("Step 2️⃣: When is Exposure Measured?")

temporality = st.selectbox(
    "Relative to outcome...",
    [
        "Before outcome occurs (forward in time)",
        "At same time (single snapshot)",
        "After outcome occurred (looking backward)"
    ]
)

# Determine implied design
if temporality == "Before outcome occurs (forward in time)":
    implied_design = "Cohort"
elif temporality == "At same time (single snapshot)":
    implied_design = "Cross-sectional"
else:
    implied_design = "Case-Control"

st.info(f"Implied Study Design: {implied_design}")

# ==========================================================
# STEP 3: PREDICTOR
# ==========================================================

st.subheader("Step 3️⃣: Predictor (Exposure) Type")

predictor_type = st.selectbox(
    "Exposure variable type:",
    ["Binary (2 groups)", "Categorical (>2 groups)", "Continuous"]
)

# ==========================================================
# DECISION LOGIC
# ==========================================================

measure = None
analysis = None
limitations = []
show_table = False

# BINARY OUTCOME
if outcome_type == "Binary":

    if implied_design == "Cohort":
        if predictor_type == "Binary (2 groups)":
            measure = "Risk Ratio (RR)"
            analysis = "Chi-square"
            show_table = True
        else:
            measure = "Adjusted Odds Ratio"
            analysis = "Logistic Regression"

    elif implied_design == "Case-Control":
        measure = "Odds Ratio (OR)"
        analysis = "Chi-square or Logistic Regression"
        show_table = True
        limitations = [
            "Cannot calculate incidence",
            "Cannot calculate risk",
            "Odds Ratio approximates Risk Ratio only when disease is rare"
        ]

    elif implied_design == "Cross-sectional":
        measure = "Prevalence Ratio or Prevalence Odds Ratio"
        analysis = "Chi-square or Logistic Regression"
        show_table = True
        limitations = [
            "Cannot determine temporality",
            "Cannot calculate incidence"
        ]

# CONTINUOUS OUTCOME
elif outcome_type == "Continuous":

    if predictor_type == "Binary (2 groups)":
        measure = "Mean Difference"
        analysis = "Independent samples t-test"
    elif predictor_type == "Categorical (>2 groups)":
        measure = "Difference in Means"
        analysis = "ANOVA"
    else:
        measure = "Beta Coefficient"
        analysis = "Linear Regression"

# RATE OUTCOME
elif outcome_type == "Rate (person-time)":

    measure = "Rate Ratio"
    analysis = "Poisson Regression"
    limitations = ["Requires person-time denominator"]

# ==========================================================
# OUTPUT
# ==========================================================

st.divider()

col1, col2 = st.columns(2)
col1.success(f"Measure: {measure}")
col2.info(f"Analysis: {analysis}")

# 2x2 table
if show_table:
    st.subheader("📊 2×2 Table Structure")
    table = pd.DataFrame(
        [["a", "b"],
         ["c", "d"]],
        columns=["Outcome +", "Outcome -"],
        index=["Exposed", "Unexposed"]
    )
    st.table(table)

# ==========================================================
# INTERPRETATION GENERATOR
# ==========================================================

st.subheader("📖 Interpretation Generator")

estimate = st.number_input("Enter your calculated estimate (optional)", value=0.0)

if estimate != 0:

    if measure in ["Risk Ratio (RR)", "Odds Ratio (OR)", "Adjusted Odds Ratio"]:
        if estimate > 1:
            st.success(f"The exposed group has {round(estimate,2)} times higher odds/risk compared to the unexposed group.")
        elif estimate < 1:
            st.success(f"The exposure appears protective (estimate = {round(estimate,2)}).")
        else:
            st.success("No association (estimate ≈ 1).")

    elif measure == "Rate Ratio":
        st.success(f"The incidence rate in the exposed group is {round(estimate,2)} times the rate in the unexposed group.")

    elif measure in ["Mean Difference", "Difference in Means"]:
        st.success(f"The average outcome differs by {round(estimate,2)} units between groups.")

# ==========================================================
# CONFOUNDING CHECK
# ==========================================================

with st.expander("🔎 Confounding Check"):
    st.markdown("""
    Ask yourself:
    - Is there a third variable associated with exposure?
    - Is it associated with outcome?
    - Is it NOT on the causal pathway?

    If YES → consider stratification or regression adjustment.
    """)

# ==========================================================
# LIMITATIONS PANEL
# ==========================================================

if limitations:
    with st.expander("⚠ What Cannot Be Estimated"):
        for item in limitations:
            st.markdown(f"- {item}")

st.markdown("---")
st.markdown("Strong epidemiologists think about design before calculation.")
