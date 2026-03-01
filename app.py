import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.stats import chi2_contingency, fisher_exact

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

st.title("🧭 Epidemiology Decision Simulator")
st.markdown("Study Design → Outcome Type → Exposure Type → Measure → Inference")

# ==========================================================
# STUDY DESIGN
# ==========================================================

st.subheader("Step 1️⃣: Study Design")

design = st.selectbox(
    "Select study design:",
    ["Cohort", "Case-Control", "Cross-sectional"]
)

# ==========================================================
# OUTCOME TYPE
# ==========================================================

st.subheader("Step 2️⃣: Outcome Variable Type")

outcome_type = st.selectbox(
    "Select outcome type:",
    [
        "Binary",
        "Categorical (Nominal >2 levels)",
        "Ordinal",
        "Continuous",
        "Rate (person-time)"
    ]
)

# ==========================================================
# EXPOSURE TYPE
# ==========================================================

st.subheader("Step 3️⃣: Exposure Variable Type")

exposure_type = st.selectbox(
    "Select exposure type:",
    [
        "Binary (2 groups)",
        "Categorical (>2 groups)",
        "Continuous"
    ]
)

# ==========================================================
# MEASURE LOGIC
# ==========================================================

st.divider()
st.subheader("Measure Based on Design and Outcome")

measure = None
analysis = None
allow_inference = False

# ==========================================================
# BINARY OUTCOME — STANDARD 2×2 LAYOUT
# ==========================================================

if outcome_type == "Binary":

    st.markdown("### Enter 2×2 Table Counts")

    st.markdown("####               Outcome +        Outcome -")

    col_header1, col_header2, col_header3 = st.columns([1,1,1])

    with col_header2:
        st.markdown("**Outcome +**")

    with col_header3:
        st.markdown("**Outcome -**")

    # Row 1: Exposed
    row1_label, row1_col1, row1_col2 = st.columns([1,1,1])

    with row1_label:
        st.markdown("**Exposed**")

    with row1_col1:
        a = st.number_input("a", min_value=0, key="a_cell")

    with row1_col2:
        b = st.number_input("b", min_value=0, key="b_cell")

    # Row 2: Unexposed
    row2_label, row2_col1, row2_col2 = st.columns([1,1,1])

    with row2_label:
        st.markdown("**Unexposed**")

    with row2_col1:
        c = st.number_input("c", min_value=0, key="c_cell")

    with row2_col2:
        d = st.number_input("d", min_value=0, key="d_cell")

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

    # -------------------------
    # INFERENCE
    # -------------------------

    if grand_total > 0 and row1_total > 0 and row2_total > 0:

        if any(x < 5 for x in [a, b, c, d]):
            st.warning("⚠ Small cell counts detected. Fisher’s Exact Test recommended.")

        risk_exp = a / row1_total if row1_total > 0 else np.nan
        risk_unexp = c / row2_total if row2_total > 0 else np.nan

        rr = risk_exp / risk_unexp if risk_unexp > 0 else np.nan
        rd = risk_exp - risk_unexp if not np.isnan(risk_exp) else np.nan
        or_val = (a * d) / (b * c) if b > 0 and c > 0 else np.nan

        st.subheader("📈 Measures")

        if not np.isnan(rr):
            st.success(f"Risk Ratio (RR) = {round(rr,3)}")
            st.success(f"Risk Difference (RD) = {round(rd,3)}")

        if not np.isnan(or_val):
            st.success(f"Odds Ratio (OR) = {round(or_val,3)}")

        chi2, p_chi, _, _ = chi2_contingency([[a,b],[c,d]])
        _, p_fisher = fisher_exact([[a,b],[c,d]])

        st.info(f"Chi-square p-value = {round(p_chi,4)}")
        st.info(f"Fisher’s Exact p-value = {round(p_fisher,4)}")

# --------------------------
# CATEGORICAL OUTCOME
# --------------------------

elif outcome_type == "Categorical (Nominal >2 levels)":

    measure = "Relative Risk Ratios or Odds Ratios"
    analysis = "Multinomial Logistic Regression"
    allow_inference = False

# --------------------------
# ORDINAL OUTCOME
# --------------------------

elif outcome_type == "Ordinal":

    measure = "Proportional Odds Ratio"
    analysis = "Ordinal Logistic Regression"
    allow_inference = False

# --------------------------
# CONTINUOUS OUTCOME
# --------------------------

elif outcome_type == "Continuous":

    if exposure_type == "Binary (2 groups)":
        measure = "Mean Difference"
        analysis = "Independent samples t-test"
    else:
        measure = "Regression Coefficient (Beta)"
        analysis = "Linear Regression"
    allow_inference = False  # Keeping 2x2 inference only for binary

# --------------------------
# RATE OUTCOME
# --------------------------

elif outcome_type == "Rate (person-time)":

    measure = "Rate Ratio"
    analysis = "Poisson Regression"
    allow_inference = True

st.success(f"Primary Measure: {measure}")
st.info(f"Primary Analysis Approach: {analysis}")

# ==========================================================
# INFERENCE SECTION (ONLY WHEN APPROPRIATE)
# ==========================================================

if allow_inference:

    st.divider()
    st.header("📊 Data Entry & Automatic Inference")

    # ==========================
    # BINARY OUTCOME
    # ==========================

    if outcome_type == "Binary":

        st.markdown("### Enter 2×2 Table Counts")

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
                st.warning("⚠ Small cell counts detected. Fisher’s Exact Test recommended.")

            risk_exp = a / row1_total if row1_total > 0 else np.nan
            risk_unexp = c / row2_total if row2_total > 0 else np.nan

            rr = risk_exp / risk_unexp if risk_unexp > 0 else np.nan
            rd = risk_exp - risk_unexp if not np.isnan(risk_exp) else np.nan
            or_val = (a * d) / (b * c) if b > 0 and c > 0 else np.nan

            st.subheader("📈 Measures")

            if not np.isnan(rr):
                st.success(f"Risk Ratio (RR) = {round(rr,3)}")
                st.success(f"Risk Difference (RD) = {round(rd,3)}")

            if not np.isnan(or_val):
                st.success(f"Odds Ratio (OR) = {round(or_val,3)}")

            # Tests
            chi2, p_chi, _, _ = chi2_contingency([[a,b],[c,d]])
            _, p_fisher = fisher_exact([[a,b],[c,d]])

            st.info(f"Chi-square p-value = {round(p_chi,4)}")
            st.info(f"Fisher’s Exact p-value = {round(p_fisher,4)}")

    # ==========================
    # RATE OUTCOME
    # ==========================

    elif outcome_type == "Rate (person-time)":

        st.markdown("### Enter Cases and Person-Time")

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

        st.success(f"Rate Ratio = {round(rr,3)}")

# ==========================================================
# CONFOUNDING CHECK
# ==========================================================

with st.expander("🔎 Confounding Check (Interactive)"):
    st.markdown("All three must be YES to meet confounding criteria.")

    c1 = st.selectbox("Associated with exposure?", ["Select", "Yes", "No"])
    c2 = st.selectbox("Associated with outcome?", ["Select", "Yes", "No"])
    c3 = st.selectbox("Not on causal pathway?", ["Select", "Yes", "No"])

    if c1 != "Select" and c2 != "Select" and c3 != "Select":
        if c1 == "Yes" and c2 == "Yes" and c3 == "Yes":
            st.warning("Likely confounder → Consider adjustment.")
        elif c3 == "No":
            st.info("Likely mediator, not confounder.")
        else:
            st.success("Does not meet confounding criteria.")

st.markdown("---")
st.markdown("Strong epidemiologists think structurally before computing.")

