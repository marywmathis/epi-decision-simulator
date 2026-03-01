import streamlit as st
import pandas as pd

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

# ==========================================================
# STYLING
# ==========================================================

st.markdown("""
<style>
.big-title {font-size:32px !important; font-weight:700;}
.section-header {font-size:22px !important; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🧭 Epidemiology Decision Simulator</p>', unsafe_allow_html=True)
st.markdown("Work through the logic: **Outcome → Type → Study Design → Measure → Formula → Test**")

# ==========================================================
# MODE SELECTOR
# ==========================================================

mode = st.radio("Mode", ["Decision Builder", "Scenario Practice", "Calculator Mode"], horizontal=True)

# ==========================================================
# SCENARIO MODE
# ==========================================================

if mode == "Scenario Practice":

    st.markdown('<p class="section-header">📝 Practice Scenario</p>', unsafe_allow_html=True)

    scenarios = {
        "Asthma & Vaping (Cohort)": 
        "Researchers follow adolescents for 5 years to see whether vaping predicts new asthma diagnoses.",

        "Smoking & Lung Cancer (Case-Control)": 
        "Researchers identify lung cancer cases and controls, then look backward to assess smoking history.",

        "Hypertension Survey (Cross-sectional)": 
        "A one-time survey measures the proportion of adults who currently have hypertension.",

        "Medication Trial (Continuous Outcome)": 
        "A study compares average systolic blood pressure between treatment and placebo groups."
    }

    selected = st.selectbox("Choose a scenario", list(scenarios.keys()))
    st.info(scenarios[selected])
    st.divider()

# ==========================================================
# DECISION BUILDER
# ==========================================================

if mode in ["Decision Builder", "Scenario Practice"]:

    st.markdown('<p class="section-header">Step 1️⃣: Outcome</p>', unsafe_allow_html=True)

    outcome = st.selectbox(
        "What is the outcome variable?",
        [
            "Disease status (Yes/No)",
            "Incidence (new cases)",
            "Prevalence (existing cases)",
            "Incidence rate (person-time)",
            "Continuous measurement"
        ]
    )

    st.markdown('<p class="section-header">Step 2️⃣: Outcome Type</p>', unsafe_allow_html=True)

    outcome_type = st.selectbox(
        "What type of variable is it?",
        [
            "Binary",
            "Rate",
            "Continuous",
            "Matched pairs",
            "Time-based comparison"
        ]
    )

    st.markdown('<p class="section-header">Step 3️⃣: Study Design</p>', unsafe_allow_html=True)

    design = st.selectbox(
        "What is the study design?",
        [
            "Cohort",
            "Cross-sectional",
            "Case-Control",
            "Matched Case-Control",
            "Case-Crossover"
        ]
    )

    measure = None
    formula = None
    test = None
    show_table = False
    warning = None

    # LOGIC ENGINE

    if design == "Cohort":
        if outcome == "Disease status (Yes/No)" and outcome_type == "Binary":
            measure = "Risk Ratio (RR)"
            formula = "RR = [a/(a+b)] / [c/(c+d)]"
            test = "Chi-square"
            show_table = True
        elif outcome == "Incidence rate (person-time)" and outcome_type == "Rate":
            measure = "Rate Ratio"
            formula = "IRR = IR exposed / IR unexposed"
            test = "Poisson regression"
        elif outcome == "Continuous measurement" and outcome_type == "Continuous":
            measure = "Mean Difference"
            formula = "Mean₁ − Mean₂"
            test = "Independent t-test"
        else:
            warning = "⚠️ That combination does not align with a standard cohort analysis."

    elif design == "Cross-sectional":
        if outcome == "Prevalence (existing cases)" and outcome_type == "Binary":
            measure = "Prevalence Ratio (PR)"
            formula = "PR = Prevalence exposed / Prevalence unexposed"
            test = "Chi-square"
            show_table = True
        else:
            warning = "⚠️ Cross-sectional studies measure prevalence."

    elif design == "Case-Control":
        if outcome_type == "Binary":
            measure = "Odds Ratio (OR)"
            formula = "OR = (a×d)/(b×c)"
            test = "Chi-square"
            show_table = True
        else:
            warning = "⚠️ Case-control studies estimate odds."

    elif design == "Matched Case-Control":
        if outcome_type == "Matched pairs":
            measure = "Matched OR"
            formula = "OR = b/c (discordant pairs)"
            test = "McNemar test"
        else:
            warning = "⚠️ Matched design requires matched data."

    elif design == "Case-Crossover":
        if outcome_type == "Time-based comparison":
            measure = "Odds Ratio"
            formula = "OR = b/c"
            test = "McNemar test"
        else:
            warning = "⚠️ Case-crossover compares exposure across time."

    st.divider()

    if measure:
        col1, col2, col3 = st.columns(3)
        col1.success(f"Measure: {measure}")
        col2.info(f"Formula: {formula}")
        col3.warning(f"Test: {test}")

    if warning:
        st.error(warning)

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
# CALCULATOR MODE
# ==========================================================

if mode == "Calculator Mode":

    st.markdown('<p class="section-header">🧮 Association Calculator</p>', unsafe_allow_html=True)

    calc_type = st.selectbox("Choose measure to calculate", ["Risk Ratio", "Odds Ratio", "Mean Difference"])

    if calc_type in ["Risk Ratio", "Odds Ratio"]:

        a = st.number_input("a", min_value=0)
        b = st.number_input("b", min_value=0)
        c = st.number_input("c", min_value=0)
        d = st.number_input("d", min_value=0)

        if st.button("Calculate"):

            if calc_type == "Risk Ratio":
                rr = (a/(a+b)) / (c/(c+d)) if (a+b)>0 and (c+d)>0 else None
                st.success(f"Risk Ratio = {round(rr,3)}")

            if calc_type == "Odds Ratio":
                or_val = (a*d)/(b*c) if b>0 and c>0 else None
                st.success(f"Odds Ratio = {round(or_val,3)}")

    if calc_type == "Mean Difference":
        mean1 = st.number_input("Mean Group 1")
        mean2 = st.number_input("Mean Group 2")

        if st.button("Calculate Difference"):
            st.success(f"Mean Difference = {round(mean1-mean2,3)}")

# ==========================================================
# EDUCATION PANEL
# ==========================================================

with st.expander("🚫 Common Mistakes"):
    st.markdown("""
    - RR cannot be calculated in case-control studies.
    - Cross-sectional studies measure prevalence.
    - Matched designs require McNemar.
    - Odds ≠ Risk.
    """)

st.markdown("---")
st.markdown("Outcome → Outcome Type → Study Design → Measure → Formula → Test")