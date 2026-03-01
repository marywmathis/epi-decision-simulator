import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.stats import chi2_contingency, fisher_exact

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

st.divider()
st.header("📊 Data Entry & Automatic Inference")

# ==========================================================
# BINARY OUTCOME
# ==========================================================

if outcome_type == "Binary":

    st.markdown("### Enter counts")

    col1, col2 = st.columns(2)

    with col1:
        a = st.number_input("Exposed & Outcome +", min_value=0)
        b = st.number_input("Exposed & Outcome -", min_value=0)

    with col2:
        c = st.number_input("Unexposed & Outcome +", min_value=0)
        d = st.number_input("Unexposed & Outcome -", min_value=0)

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

        # Small cell warning
        if any(x < 5 for x in [a, b, c, d]):
            st.warning("⚠ Some cells have counts <5. Fisher's Exact Test is recommended.")

        # Risks
        risk_exp = a / row1_total if row1_total > 0 else np.nan
        risk_unexp = c / row2_total if row2_total > 0 else np.nan

        rr = risk_exp / risk_unexp if risk_unexp > 0 else np.nan
        rd = risk_exp - risk_unexp if not np.isnan(risk_exp) and not np.isnan(risk_unexp) else np.nan
        ar = rd
        afe = (rr - 1) / rr if rr > 0 else np.nan

        or_val = (a * d) / (b * c) if b > 0 and c > 0 else np.nan

        st.subheader("📈 Measures of Association")

        if not np.isnan(rr):
            st.success(f"Risk Ratio (RR) = {round(rr,3)}")

        if not np.isnan(or_val):
            st.success(f"Odds Ratio (OR) = {round(or_val,3)}")

        if not np.isnan(rd):
            st.success(f"Risk Difference (RD) = {round(rd,3)}")

        if not np.isnan(afe):
            st.success(f"Attributable Fraction in Exposed (AFE) = {round(afe,3)}")

        # Tests
        chi2, p_chi, _, _ = chi2_contingency([[a,b],[c,d]])
        _, p_fisher = fisher_exact([[a,b],[c,d]])

        st.info(f"Chi-square p-value = {round(p_chi,4)}")
        st.info(f"Fisher's Exact p-value = {round(p_fisher,4)}")

        # Confidence intervals
        if a>0 and b>0 and c>0 and d>0:

            # RR CI
            se_log_rr = math.sqrt((1/a)-(1/row1_total)+(1/c)-(1/row2_total))
            ci_low_rr = math.exp(math.log(rr)-1.96*se_log_rr)
            ci_high_rr = math.exp(math.log(rr)+1.96*se_log_rr)

            # OR CI
            se_log_or = math.sqrt(1/a+1/b+1/c+1/d)
            ci_low_or = math.exp(math.log(or_val)-1.96*se_log_or)
            ci_high_or = math.exp(math.log(or_val)+1.96*se_log_or)

            st.info(f"95% CI for RR: ({round(ci_low_rr,3)}, {round(ci_high_rr,3)})")
            st.info(f"95% CI for OR: ({round(ci_low_or,3)}, {round(ci_high_or,3)})")

            # CI interpretation
            if ci_low_rr <= 1 <= ci_high_rr:
                st.warning("RR CI includes 1 → Not statistically significant at α=0.05.")
            else:
                st.success("RR CI does NOT include 1 → Statistically significant at α=0.05.")

        if design == "Case-Control":
            st.info("Note: OR approximates RR only when disease is rare.")

# ==========================================================
# RATE OUTCOME
# ==========================================================

elif outcome_type == "Rate (person-time)":

    st.markdown("### Enter cases and person-time")

    col1, col2 = st.columns(2)

    with col1:
        cases1 = st.number_input("Cases (Exposed)", min_value=0)
        py1 = st.number_input("Person-Time (Exposed)", min_value=1)

    with col2:
        cases2 = st.number_input("Cases (Unexposed)", min_value=0)
        py2 = st.number_input("Person-Time (Unexposed)", min_value=1)

    ir1 = cases1 / py1
    ir2 = cases2 / py2
    rr = ir1 / ir2 if ir2 > 0 else np.nan

    table = pd.DataFrame(
        [[cases1, py1], [cases2, py2]],
        columns=["Cases", "Person-Time"],
        index=["Exposed", "Unexposed"]
    )

    st.subheader("Rate Table")
    st.table(table)

    if not np.isnan(rr):

        st.success(f"Rate Ratio = {round(rr,3)}")

        if cases1 > 0 and cases2 > 0:
            se_log_rr = math.sqrt((1/cases1)+(1/cases2))
            ci_low = math.exp(math.log(rr)-1.96*se_log_rr)
            ci_high = math.exp(math.log(rr)+1.96*se_log_rr)

            st.info(f"95% CI: ({round(ci_low,3)}, {round(ci_high,3)})")

            if ci_low <= 1 <= ci_high:
                st.warning("CI includes 1 → Not statistically significant.")
            else:
                st.success("CI excludes 1 → Statistically significant.")

# ==========================================================
# INTERPRETATION
# ==========================================================

st.divider()
st.header("📖 Interpretation")

if outcome_type == "Binary" and grand_total > 0:

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
            st.info("Likely mediator, not confounder.")
        else:
            st.success("Does not meet criteria for confounding.")

st.markdown("---")
st.markdown("Strong epidemiologists think about design before inference.")
