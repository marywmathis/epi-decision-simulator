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

# ----------------------
# BINARY OUTCOME
# ----------------------

if outcome_type == "Binary":

    if implied_design == "Cohort":

        if predictor_type == "Binary (2 groups)":
            measure = "Risk Ratio (RR)"
            analysis = "Chi-square (crude) or Logistic Regression (adjusted)"
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

# ----------------------
# CONTINUOUS OUTCOME
# ----------------------

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

# ----------------------
# RATE OUTCOME
# ----------------------

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

# 2×2 table display
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
# INTERPRETATION GENERATOR (IMPROVED)
# ==========================================================

st.subheader("📖 Interpretation Generator")

st.markdown("""
**What is a calculated estimate?**

This is the numerical value you computed (for example:  
Risk Ratio = 2.3, Odds Ratio = 0.75, Rate Ratio = 1.8, Mean Difference = 5.2).

Enter your value below to generate correct epidemiologic wording.
""")

measure_type = st.selectbox(
    "Which measure did you calculate?",
    ["Risk Ratio (RR)", "Odds Ratio (OR)", "Rate Ratio (IRR)", "Mean Difference"]
)

estimate = st.number_input(
    "Enter your calculated value",
    min_value=-1000.0,
    value=0.0
)

if estimate != 0:

    st.markdown("### Interpretation:")

    if measure_type == "Risk Ratio (RR)":

        if estimate > 1:
            st.success(f"The exposed group has {round(estimate,2)} times the risk compared to the unexposed group.")
        elif estimate < 1:
            st.success(f"The exposure appears protective. The risk in the exposed group is {round(estimate,2)} times that of the unexposed group.")
        else:
            st.success("An RR of 1 suggests no association.")

    elif measure_type == "Odds Ratio (OR)":

        if estimate > 1:
            st.success(f"The odds of outcome in the exposed group are {round(estimate,2)} times the odds in the unexposed group.")
        elif estimate < 1:
            st.success(f"The exposure appears protective (OR = {round(estimate,2)}).")
        else:
            st.success("An OR of 1 suggests no association.")

    elif measure_type == "Rate Ratio (IRR)":
        st.success(f"The incidence rate in the exposed group is {round(estimate,2)} times the rate in the unexposed group.")

    elif measure_type == "Mean Difference":
        st.success(f"The average outcome differs by {round(estimate,2)} units between groups.")

    st.info("Remember: Statistical significance depends on confidence intervals or p-values, not just the point estimate.")

# ==========================================================
# INTERACTIVE CONFOUNDING CHECK
# ==========================================================

with st.expander("🔎 Confounding Check (Interactive)"):

    st.markdown("A variable is a confounder only if ALL three conditions are TRUE:")

    q1 = st.selectbox(
        "1️⃣ Is the third variable associated with the exposure?",
        ["Select", "Yes", "No"]
    )

    q2 = st.selectbox(
        "2️⃣ Is it associated with the outcome?",
        ["Select", "Yes", "No"]
    )

    q3 = st.selectbox(
        "3️⃣ Is it NOT on the causal pathway between exposure and outcome?",
        ["Select", "Yes", "No"]
    )

    if q1 != "Select" and q2 != "Select" and q3 != "Select":

        if q1 == "Yes" and q2 == "Yes" and q3 == "Yes":
            st.warning("This variable meets criteria for confounding. Consider stratification or regression adjustment.")

        elif q3 == "No":
            st.info("If it lies on the causal pathway, it is a mediator — not a confounder.")

        else:
            st.success("This variable does NOT meet criteria for confounding.")

# ==========================================================
# LIMITATIONS PANEL
# ==========================================================

if limitations:
    with st.expander("⚠ What Cannot Be Estimated in This Design"):
        for item in limitations:
            st.markdown(f"- {item}")

st.markdown("---")
st.markdown("Strong epidemiologists think about design before calculation.")
