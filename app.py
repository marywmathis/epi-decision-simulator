import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

st.title("🧭 Epidemiology Decision Simulator")
st.markdown("Outcome → Time → Design → Predictor → Measure → Inference → Interpretation")

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
# MEASURE LOGIC DISPLAY
# ==========================================================

st.divider()
st.subheader("Measure Based on Design")

if outcome_type == "Binary":
    if design == "Cohort":
        measure = "Risk Ratio (RR)"
    elif design == "Case-Control":
        measure = "Odds Ratio (OR)"
    else:
        measure = "Prevalence Ratio or OR"
elif outcome_type == "Continuous":
    measure = "Mean Difference / Regression Coefficient"
elif outcome_type == "Rate (person-time)":
    measure = "Rate Ratio"

st.success(f"Primary Measure: {measure}")

# ==========================================================
# 📊 INTERACTIVE DATA ENTRY TABLE
# ==========================================================

st.divider()
st.header("📊 Data Entry & Automatic Inference")

# -------------------------
# BINARY OUTCOME (2×2 TABLE)
# -------------------------

if outcome_type == "Binary":

    st.markdown("### Enter counts for each cell")

    col1, col2 = st.columns(2)

    with col1:
        a = st.number_input("Exposed & Outcome +", min_value=0)
        b = st.number_input("Exposed & Outcome -", min_value=0)

    with col2:
        c = st.number_input("Unexposed & Outcome +", min_value=0)
        d = st.number_input("Unexposed & Outcome -", min_value=0)

    # Totals
    row1_total = a + b
    row2_total = c + d
    col1_total = a + c
    col2_total = b + d
    grand_total = row1_total + row2_total

    table = pd.DataFrame(
        [
            [a, b, row1_total],
            [c, d, row2_total],
            [col1_total, col2_total, grand_total]
        ],
        columns=["Outcome +", "Outcome -", "Row Total"],
        index=["Exposed", "Unexposed", "Column Total"]
    )

    st.subheader("2×2 Table with Totals")
    st.table(table)

    if grand_total > 0 and row1_total > 0 and row2_total > 0:

        # Risk Ratio
        rr = (a/row1_total) / (c/row2_total) if c > 0 else np.nan

        # Odds Ratio
        or_val = (a*d)/(b*c) if b > 0 and c > 0 else np.nan

        st.subheader("📈 Measures of Association")

        if not np.isnan(rr):
            st.success(f"Risk Ratio (RR) = {round(rr,3)}")

        if not np.isnan(or_val):
            st.success(f"Odds Ratio (OR) = {round(or_val,3)}")

        # Chi-square
        chi2, p, _, _ = chi2_contingency([[a,b],[c,d]])
        st.info(f"Chi-square p-value = {round(p,4)}")

        # Confidence Intervals
        if a>0 and b>0 and c>0 and d>0:

            # RR CI
            se_log_rr = math.sqrt((1/a)-(1/row1_total)+(1/c)-(1/row2_total))
            ci_low_rr = math.exp(math.log(rr)-1.96*se_log_rr)
            ci_high_rr = math.exp(math.log(rr)+1.96*se_log_rr)
            st.info(f"95% CI for RR: ({round(ci_low_rr,3)}, {round(ci_high_rr,3)})")

            # OR CI
            se_log_or = math.sqrt(1/a+1/b+1/c+1/d)
            ci_low_or = math.exp(math.log(or_val)-1.96*se_log_or)
            ci_high_or = math.exp(math.log(or_val)+1.96*se_log_or)
            st.info(f"95% CI for OR: ({round(ci_low_or,3)}, {round(ci_high_or,3)})")

# -------------------------
# RATE OUTCOME
# -------------------------

elif outcome_type == "Rate (person-time)":

    st.markdown("### Enter cases and person-time")

    col1, col2 = st.columns(2)

    with col1:
        cases1 = st.number_input("Cases (Exposed)", min_value=0)
        py1 = st.number_input("Person-Time (Exposed)", min_value=1)

    with col2:
        cases2 = st.number_input("Cases (Unexposed)", min_value=0)
        py2 = st.number_input("Person-Time (Unexposed)", min_value=1)

    ir1 = cases1/py1
    ir2 = cases2/py2
    rr = ir1/ir2 if ir2 > 0 else np.nan

    rate_table = pd.DataFrame(
        [
            [cases1, py1],
            [cases2, py2]
        ],
        columns=["Cases", "Person-Time"],
        index=["Exposed", "Unexposed"]
    )

    st.subheader("Rate Data Table")
    st.table(rate_table)

    if not np.isnan(rr):
        st.success(f"Rate Ratio = {round(rr,3)}")

        if cases1>0 and cases2>0:
            se_log_rr = math.sqrt((1/cases1)+(1/cases2))
            ci_low = math.exp(math.log(rr)-1.96*se_log_rr)
            ci_high = math.exp(math.log(rr)+1.96*se_log_rr)
            st.info(f"95% CI: ({round(ci_low,3)}, {round(ci_high,3)})")

# ==========================================================
# 📖 AUTOMATIC INTERPRETATION
# ==========================================================

st.divider()
st.header("📖 Interpretation")

if outcome_type == "Binary" and grand_total > 0 and row1_total > 0 and row2_total > 0:

    if not np.isnan(rr):
        if rr > 1:
            st.success(f"The exposed group has {round(rr,2)} times the risk compared to the unexposed group.")
        elif rr < 1:
            st.success("The exposure appears protective.")
        else:
            st.success("No association detected.")

elif outcome_type == "Rate (person-time)" and not np.isnan(rr):
    st.success(f"The incidence rate is {round(rr,2)} times higher in the exposed group.")

# ==========================================================
# 🔎 INTERACTIVE CONFOUNDING CHECK
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

st.markdown("---")
st.markdown("Strong epidemiologists think about design before inference.")
