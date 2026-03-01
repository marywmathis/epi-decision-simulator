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
st.markdown("Outcome → Outcome Type → Study Design → Measure → Formula → Test")

# ==========================================================
# MODE SELECTOR
# ==========================================================

mode = st.radio(
    "Select Mode",
    [
        "Decision Builder",
        "Scenario Practice",
        "Association Calculator",
        "Person-Time (Cohort) Calculator"
    ],
    horizontal=True
)

# ==========================================================
# SCENARIO MODE
# ==========================================================

if mode == "Scenario Practice":

    st.subheader("📝 Practice Scenario")

    scenarios = {
        "Asthma & Vaping (Cohort)":
        "Researchers follow adolescents for 5 years to assess new asthma diagnoses.",

        "Smoking & Lung Cancer (Case-Control)":
        "Investigators identify lung cancer cases and controls and assess past smoking.",

        "Hypertension Survey (Cross-sectional)":
        "A one-time survey measures the proportion currently with hypertension.",

        "Blood Pressure Trial (Continuous)":
        "Researchers compare mean systolic BP between treatment and placebo."
    }

    selected = st.selectbox("Choose scenario", list(scenarios.keys()))
    st.info(scenarios[selected])
    st.divider()

# ==========================================================
# DECISION BUILDER
# ==========================================================

if mode in ["Decision Builder", "Scenario Practice"]:

    st.subheader("Step 1️⃣: Outcome")

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

    st.subheader("Step 2️⃣: Outcome Type")

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

    st.subheader("Step 3️⃣: Study Design")

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
    warning = None
    show_table = False

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
            test = "Independent samples t-test"

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
            measure = "Matched Odds Ratio"
            formula = "OR = b/c (discordant pairs)"
            test = "McNemar test"
        else:
            warning = "⚠️ Matched designs require matched data."

    elif design == "Case-Crossover":

        if outcome_type == "Time-based comparison":
            measure = "Odds Ratio"
            formula = "OR = b/c"
            test = "McNemar test"
        else:
            warning = "⚠️ Case-crossover compares exposure over time."

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
# ASSOCIATION CALCULATOR
# ==========================================================

if mode == "Association Calculator":

    st.subheader("🧮 Association Calculator")

    calc_type = st.selectbox(
        "Choose measure",
        ["Risk Ratio", "Odds Ratio", "Mean Difference"]
    )

    if calc_type in ["Risk Ratio", "Odds Ratio"]:

        a = st.number_input("a", min_value=0.0)
        b = st.number_input("b", min_value=0.0)
        c = st.number_input("c", min_value=0.0)
        d = st.number_input("d", min_value=0.0)

        if st.button("Calculate"):

            if calc_type == "Risk Ratio":
                if (a+b)>0 and (c+d)>0:
                    rr = (a/(a+b)) / (c/(c+d))
                    st.success(f"Risk Ratio = {round(rr,4)}")

            if calc_type == "Odds Ratio":
                if b>0 and c>0:
                    or_val = (a*d)/(b*c)
                    st.success(f"Odds Ratio = {round(or_val,4)}")

    if calc_type == "Mean Difference":

        mean1 = st.number_input("Mean Group 1")
        mean2 = st.number_input("Mean Group 2")

        if st.button("Calculate Difference"):
            st.success(f"Mean Difference = {round(mean1-mean2,4)}")

# ==========================================================
# PERSON-TIME CALCULATOR
# ==========================================================

if mode == "Person-Time (Cohort) Calculator":

    st.subheader("⏳ Person-Time Calculator (Cohort Study)")

    st.markdown("Enter each participant's time at risk (in years).")

    num_people = st.number_input("Number of participants", min_value=1, value=5)

    times = []
    for i in range(int(num_people)):
        t = st.number_input(f"Participant {i+1} time (years)", min_value=0.0, key=i)
        times.append(t)

    total_person_time = sum(times)

    st.info(f"Total Person-Years = {round(total_person_time,3)}")

    st.markdown("### Incidence Rate Calculation")

    cases = st.number_input("Number of new cases", min_value=0.0)

    if st.button("Calculate Incidence Rate"):
        if total_person_time > 0:
            ir = cases / total_person_time
            st.success(f"Incidence Rate = {round(ir,4)} cases per person-year")

    st.markdown("### Rate Ratio (Two Groups)")

    st.markdown("Group 1")
    cases1 = st.number_input("Cases (Group 1)", min_value=0.0, key="g1c")
    py1 = st.number_input("Person-Years (Group 1)", min_value=0.0, key="g1py")

    st.markdown("Group 2")
    cases2 = st.number_input("Cases (Group 2)", min_value=0.0, key="g2c")
    py2 = st.number_input("Person-Years (Group 2)", min_value=0.0, key="g2py")

    if st.button("Calculate Rate Ratio"):
        if py1>0 and py2>0:
            ir1 = cases1/py1
            ir2 = cases2/py2
            rr = ir1/ir2 if ir2>0 else None
            st.success(f"Rate Ratio = {round(rr,4)}")

# ==========================================================
# EDUCATION PANEL
# ==========================================================

with st.expander("🚫 Common Mistakes"):
    st.markdown("""
    - RR cannot be calculated in case-control studies.
    - Cross-sectional studies measure prevalence.
    - Matched designs require McNemar.
    - Odds ≠ Risk.
    - Person-time stops at outcome or loss to follow-up.
    """)

st.markdown("---")
st.markdown("Always identify: Outcome → Type → Design → Measure → Test")
