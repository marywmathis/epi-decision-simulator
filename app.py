import streamlit as st
import pandas as pd

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

st.title("🧭 Epidemiology Decision Simulator")
st.markdown("Outcome → Outcome Type → Study Design → Predictor Type → Measure → Test/Model")

# ==========================================================
# MODE
# ==========================================================

mode = st.radio(
    "Select Mode",
    [
        "Decision Builder",
        "Association Calculator",
        "Person-Time Calculator"
    ],
    horizontal=True
)

# ==========================================================
# DECISION BUILDER
# ==========================================================

if mode == "Decision Builder":

    st.subheader("Step 1️⃣: Outcome Variable")

    outcome = st.selectbox(
        "What is the outcome?",
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
        "Outcome variable type",
        [
            "Binary",
            "Rate",
            "Continuous"
        ]
    )

    st.subheader("Step 3️⃣: Study Design")

    design = st.selectbox(
        "Study design",
        [
            "Cohort",
            "Cross-sectional",
            "Case-Control",
            "Matched Case-Control",
            "Case-Crossover"
        ]
    )

    st.subheader("Step 4️⃣: Predictor Variable Type")

    predictor_type = st.selectbox(
        "What type of predictor (exposure) variable?",
        [
            "Binary (2 groups)",
            "Categorical (>2 groups)",
            "Continuous",
            "Matched pairs exposure"
        ]
    )

    measure = None
    model = None
    test = None
    show_table = False
    warning = None

    # ======================================================
    # LOGIC ENGINE
    # ======================================================

    # ------------------------
    # BINARY OUTCOME
    # ------------------------

    if outcome_type == "Binary":

        if design == "Cohort":

            if predictor_type == "Binary (2 groups)":
                measure = "Risk Ratio (RR)"
                test = "Chi-square"
                show_table = True

            elif predictor_type == "Categorical (>2 groups)":
                measure = "Multiple Risk Ratios"
                test = "Chi-square (overall) or Logistic Regression"

            elif predictor_type == "Continuous":
                measure = "Odds Ratio (per unit increase)"
                model = "Logistic Regression"

        elif design == "Case-Control":

            if predictor_type == "Binary (2 groups)":
                measure = "Odds Ratio (OR)"
                test = "Chi-square"
                show_table = True

            elif predictor_type in ["Categorical (>2 groups)", "Continuous"]:
                measure = "Odds Ratio"
                model = "Logistic Regression"

        elif design == "Cross-sectional":

            if predictor_type == "Binary (2 groups)":
                measure = "Prevalence Ratio"
                test = "Chi-square"
                show_table = True

            else:
                measure = "Prevalence Odds Ratio"
                model = "Logistic Regression"

        elif design == "Matched Case-Control":

            measure = "Matched Odds Ratio"
            test = "McNemar test"

        elif design == "Case-Crossover":

            measure = "Odds Ratio"
            test = "McNemar test"

    # ------------------------
    # CONTINUOUS OUTCOME
    # ------------------------

    elif outcome_type == "Continuous":

        if predictor_type == "Binary (2 groups)":
            measure = "Mean Difference"
            test = "Independent samples t-test"

        elif predictor_type == "Categorical (>2 groups)":
            measure = "Difference in Means"
            test = "ANOVA"

        elif predictor_type == "Continuous":
            measure = "Beta coefficient"
            model = "Linear Regression"

    # ------------------------
    # RATE OUTCOME
    # ------------------------

    elif outcome_type == "Rate":

        measure = "Rate Ratio"
        model = "Poisson Regression"

    # ======================================================
    # OUTPUT
    # ======================================================

    st.divider()

    if measure:
        col1, col2, col3 = st.columns(3)

        col1.success(f"Measure: {measure}")

        if test:
            col2.info(f"Statistical Test: {test}")

        if model:
            col3.warning(f"Model: {model}")

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

    st.subheader("🧮 Risk Ratio / Odds Ratio Calculator")

    a = st.number_input("a", min_value=0.0)
    b = st.number_input("b", min_value=0.0)
    c = st.number_input("c", min_value=0.0)
    d = st.number_input("d", min_value=0.0)

    if st.button("Calculate RR and OR"):

        if (a+b)>0 and (c+d)>0 and b>0 and c>0:

            rr = (a/(a+b)) / (c/(c+d))
            or_val = (a*d)/(b*c)

            st.success(f"Risk Ratio = {round(rr,4)}")
            st.success(f"Odds Ratio = {round(or_val,4)}")

# ==========================================================
# PERSON-TIME CALCULATOR
# ==========================================================

if mode == "Person-Time Calculator":

    st.subheader("⏳ Person-Time Calculator")

    n = st.number_input("Number of participants", min_value=1, value=5)

    times = []
    for i in range(int(n)):
        t = st.number_input(f"Participant {i+1} time (years)", min_value=0.0, key=i)
        times.append(t)

    total_py = sum(times)

    st.info(f"Total Person-Years = {round(total_py,3)}")

    cases = st.number_input("Number of new cases", min_value=0.0)

    if st.button("Calculate Incidence Rate"):
        if total_py > 0:
            ir = cases / total_py
            st.success(f"Incidence Rate = {round(ir,4)} cases per person-year")

# ==========================================================
# EDUCATIONAL PANEL
# ==========================================================

with st.expander("🧠 How to Think Through This"):
    st.markdown("""
    1. Identify outcome variable.
    2. Determine if outcome is binary, continuous, or rate.
    3. Identify study design.
    4. Determine predictor variable type.
    5. Decide: table-based test or regression model?
    """)

with st.expander("🚫 Common Mistakes"):
    st.markdown("""
    - Risk Ratio cannot be calculated in case-control studies.
    - Continuous predictors require regression.
    - Multi-category predictors often require ANOVA or regression.
    - Matching changes the statistical test.
    """)

st.markdown("---")
st.markdown("Think structurally, not memorization-based.")
