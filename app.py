import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, t
import math

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

st.title("🧭 Epidemiology Decision Simulator")
st.markdown("Outcome → Time → Design → Predictor → Measure → Interpretation → Inference")

# ==========================================================
# STEP 1: OUTCOME
# ==========================================================

st.subheader("Step 1️⃣: Outcome Type")

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
    design = "Cohort"
elif temporality == "At same time (single snapshot)":
    design = "Cross-sectional"
else:
    design = "Case-Control"

st.info(f"Implied Study Design: {design}")

# ==========================================================
# STEP 3: PREDICTOR
# ==========================================================

st.subheader("Step 3️⃣: Predictor Type")

predictor_type = st.selectbox(
    "Exposure variable type:",
    ["Binary (2 groups)", "Categorical (>2 groups)", "Continuous"]
)

# ==========================================================
# MEASURE LOGIC
# ==========================================================

measure = None
analysis = None
show_table = False

if outcome_type == "Binary":

    if design == "Cohort":
        measure = "Risk Ratio (RR)"
        analysis = "Chi-square or Logistic Regression"
        show_table = True

    elif design == "Case-Control":
        measure = "Odds Ratio (OR)"
        analysis = "Chi-square or Logistic Regression"
        show_table = True

    elif design == "Cross-sectional":
        measure = "Prevalence Ratio or Odds Ratio"
        analysis = "Chi-square or Logistic Regression"
        show_table = True

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

elif outcome_type == "Rate (person-time)":

    measure = "Rate Ratio"
    analysis = "Poisson Regression"

st.divider()
st.success(f"Measure: {measure}")
st.info(f"Primary Analysis Approach: {analysis}")

# ==========================================================
# 2x2 TABLE DISPLAY
# ==========================================================

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

st.markdown("""
Enter the numeric estimate you calculated (e.g., RR = 2.3).
""")

estimate = st.number_input("Enter estimate value", value=0.0)

if estimate != 0:

    if measure in ["Risk Ratio (RR)", "Odds Ratio (OR)"]:

        if estimate > 1:
            st.success(f"The exposed group has {round(estimate,2)} times higher risk/odds compared to the unexposed group.")
        elif estimate < 1:
            st.success(f"The exposure appears protective (estimate = {round(estimate,2)}).")
        else:
            st.success("Estimate ≈ 1 suggests no association.")

    elif measure == "Mean Difference":
        st.success(f"The average outcome differs by {round(estimate,2)} units between groups.")

    elif measure == "Rate Ratio":
        st.success(f"The incidence rate is {round(estimate,2)} times higher in the exposed group.")

    st.info("Statistical significance depends on CI or p-value, not just the point estimate.")

# ==========================================================
# CONFOUNDING CHECK
# ==========================================================

with st.expander("🔎 Confounding Check (Interactive)"):

    st.markdown("All three must be YES for confounding.")

    c1 = st.selectbox("Associated with exposure?", ["Select", "Yes", "No"])
    c2 = st.selectbox("Associated with outcome?", ["Select", "Yes", "No"])
    c3 = st.selectbox("Not on causal pathway?", ["Select", "Yes", "No"])

    if c1 != "Select" and c2 != "Select" and c3 != "Select":

        if c1 == "Yes" and c2 == "Yes" and c3 == "Yes":
            st.warning("This variable is likely a confounder. Consider adjustment.")
        elif c3 == "No":
            st.info("This is likely a mediator, not a confounder.")
        else:
            st.success("Does not meet criteria for confounding.")

# ==========================================================
# 🔬 INFERENCE MODULE (LAST STEP)
# ==========================================================

st.divider()
st.header("🔬 Final Step: Statistical Inference")

st.markdown("Enter raw data to calculate estimate, 95% CI, and p-value.")

# -------------------------
# 2x2 Inference
# -------------------------

if outcome_type == "Binary":

    a = st.number_input("a (Exposed +)", min_value=0)
    b = st.number_input("b (Exposed -)", min_value=0)
    c = st.number_input("c (Unexposed +)", min_value=0)
    d = st.number_input("d (Unexposed -)", min_value=0)

    if st.button("Run 2×2 Analysis"):

        table = np.array([[a, b], [c, d]])

        # Chi-square
        chi2, p, _, _ = chi2_contingency(table)

        # Risk Ratio
        if (a+b)>0 and (c+d)>0:
            rr = (a/(a+b)) / (c/(c+d))
            se_log_rr = math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+d)))
            ci_low = math.exp(math.log(rr)-1.96*se_log_rr)
            ci_high = math.exp(math.log(rr)+1.96*se_log_rr)
            st.success(f"Risk Ratio = {round(rr,3)}")
            st.success(f"95% CI: ({round(ci_low,3)}, {round(ci_high,3)})")

        # Odds Ratio
        if b>0 and c>0:
            or_val = (a*d)/(b*c)
            se_log_or = math.sqrt(1/a+1/b+1/c+1/d)
            ci_low_or = math.exp(math.log(or_val)-1.96*se_log_or)
            ci_high_or = math.exp(math.log(or_val)+1.96*se_log_or)
            st.success(f"Odds Ratio = {round(or_val,3)}")
            st.success(f"95% CI: ({round(ci_low_or,3)}, {round(ci_high_or,3)})")

        st.info(f"Chi-square p-value = {round(p,4)}")

# -------------------------
# Continuous Outcome
# -------------------------

elif outcome_type == "Continuous":

    mean1 = st.number_input("Mean (Group 1)")
    sd1 = st.number_input("SD (Group 1)")
    n1 = st.number_input("n (Group 1)", min_value=1)

    mean2 = st.number_input("Mean (Group 2)")
    sd2 = st.number_input("SD (Group 2)")
    n2 = st.number_input("n (Group 2)", min_value=1)

    if st.button("Run t-test"):

        diff = mean1 - mean2
        se = math.sqrt((sd1**2/n1)+(sd2**2/n2))
        df = n1+n2-2
        t_stat = diff/se
        p = 2*(1-t.cdf(abs(t_stat), df))
        ci_low = diff - 1.96*se
        ci_high = diff + 1.96*se

        st.success(f"Mean Difference = {round(diff,3)}")
        st.success(f"95% CI: ({round(ci_low,3)}, {round(ci_high,3)})")
        st.info(f"p-value = {round(p,4)}")

# -------------------------
# Rate Outcome
# -------------------------

elif outcome_type == "Rate (person-time)":

    cases1 = st.number_input("Cases (Exposed)", min_value=0)
    py1 = st.number_input("Person-years (Exposed)", min_value=1)

    cases2 = st.number_input("Cases (Unexposed)", min_value=0)
    py2 = st.number_input("Person-years (Unexposed)", min_value=1)

    if st.button("Run Rate Analysis"):

        ir1 = cases1/py1
        ir2 = cases2/py2
        rr = ir1/ir2
        se_log_rr = math.sqrt((1/cases1)+(1/cases2))
        ci_low = math.exp(math.log(rr)-1.96*se_log_rr)
        ci_high = math.exp(math.log(rr)+1.96*se_log_rr)

        st.success(f"Rate Ratio = {round(rr,3)}")
        st.success(f"95% CI: ({round(ci_low,3)}, {round(ci_high,3)})")

st.markdown("---")
st.markdown("Strong epidemiologists think about design before inference.")
