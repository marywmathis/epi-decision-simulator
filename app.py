import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import math

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

# ==========================================================
# CI VISUALIZATION HELPER
# ==========================================================

def draw_ci(label, estimate, ci_low, ci_high):
    """Draw a CI number line using HTML — no extra libraries required."""

    significant = not (ci_low <= 1 <= ci_high)
    color = "#2e7d32" if significant else "#c0392b"
    sig_text = "CI does not cross 1 → statistically significant" if significant else "CI crosses 1 → not statistically significant"

    all_vals = [ci_low, ci_high, estimate, 1.0]
    span = max(all_vals) - min(all_vals)
    pad = max(span * 0.35, 0.4)
    x_min = max(0.001, min(all_vals) - pad)
    x_max = max(all_vals) + pad

    def to_pct(val):
        return round((val - x_min) / (x_max - x_min) * 100, 2)

    pct_low  = to_pct(ci_low)
    pct_high = to_pct(ci_high)
    pct_est  = to_pct(estimate)
    pct_null = to_pct(1.0)

    html = f"""
    <div style="background:#f9f9f9; border-radius:6px; padding:16px 12px 8px 12px; margin:8px 0 16px 0;">
      <div style="position:relative; height:60px; margin: 0 20px;">
        <div style="position:absolute; top:28px; left:0; right:0; height:2px; background:#cccccc;"></div>
        <div style="position:absolute; top:24px; left:{pct_low}%; width:{pct_high - pct_low}%;
                    height:10px; background:{color}; border-radius:5px;"></div>
        <div style="position:absolute; top:20px; left:calc({pct_est}% - 9px);
                    width:18px; height:18px; background:{color}; border-radius:50%;"></div>
        <div style="position:absolute; top:8px; left:{pct_null}%; width:2px; height:44px;
                    background:#333; border-left: 2px dashed #333;"></div>
        <div style="position:absolute; top:0px; left:{pct_null}%;
                    transform:translateX(-50%); font-size:11px; color:#333; white-space:nowrap;">1 (null)</div>
        <div style="position:absolute; top:46px; left:{pct_low}%;
                    transform:translateX(-50%); font-size:11px; color:{color};">{round(ci_low,2)}</div>
        <div style="position:absolute; top:46px; left:{pct_est}%;
                    transform:translateX(-50%); font-size:12px; color:{color};
                    font-weight:bold; white-space:nowrap;">{label} = {round(estimate,2)}</div>
        <div style="position:absolute; top:46px; left:{pct_high}%;
                    transform:translateX(-50%); font-size:11px; color:{color};">{round(ci_high,2)}</div>
      </div>
      <div style="text-align:center; font-size:12px; color:{color}; font-style:italic; margin-top:28px;">
        {sig_text}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ==========================================================
# APP HEADER + TABS
# ==========================================================

st.title("🧭 Epidemiology Decision Simulator")
st.markdown("Study Design → Outcome Type → Exposure Type → Table → Run Analysis → Interpretation")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Measures of Association", "📐 Advanced Epi Measures", "📏 Standardization", "🎯 Practice on Your Own"])

# ==========================================================
# TAB 1: MEASURES OF ASSOCIATION (original app)
# ==========================================================


with tab1:

    # ----------------------------------------------------------
    # PRESET SCENARIOS
    # ----------------------------------------------------------

    PRESETS = {
        "None — I'll enter my own data": None,
        "Cohort: Smoking & Lung Cancer": {
            "design": "Cohort", "outcome_type": "Binary", "exposure_type": "Binary (2 groups)",
            "row_0": "Smoker", "row_1": "Non-smoker",
            "col_0": "Lung Cancer", "col_1": "No Lung Cancer",
            "cell_0_0": 84, "cell_0_1": 2916, "cell_1_0": 14, "cell_1_1": 2986,
            "description": (
                "**Scenario:** A prospective cohort follows 6,000 adults over 10 years. "
                "Smokers develop lung cancer at a substantially higher rate than non-smokers. "
                "*Adapted from Doll & Hill (1950), British Doctors Study.*"
            )
        },
        "Case-Control: H. pylori & Gastric Ulcer": {
            "design": "Case-Control", "outcome_type": "Binary", "exposure_type": "Binary (2 groups)",
            "row_0": "H. pylori positive", "row_1": "H. pylori negative",
            "col_0": "Gastric Ulcer (Case)", "col_1": "No Ulcer (Control)",
            "cell_0_0": 118, "cell_0_1": 62, "cell_1_0": 32, "cell_1_1": 138,
            "description": (
                "**Scenario:** A hospital-based case-control study recruits 150 patients with confirmed "
                "gastric ulcer and 200 controls. Past H. pylori infection assessed via serology. "
                "*Adapted from Marshall & Warren (1984).*"
            )
        },
        "Cross-sectional: Obesity & Hypertension": {
            "design": "Cross-sectional", "outcome_type": "Binary", "exposure_type": "Binary (2 groups)",
            "row_0": "Obese (BMI ≥ 30)", "row_1": "Non-obese (BMI < 30)",
            "col_0": "Hypertension", "col_1": "No Hypertension",
            "cell_0_0": 210, "cell_0_1": 290, "cell_1_0": 120, "cell_1_1": 880,
            "description": (
                "**Scenario:** A one-time cross-sectional health survey of 1,500 adults measures "
                "current BMI and blood pressure simultaneously. *Adapted from NHANES survey data.*"
            )
        },
        "Cohort: Asbestos Exposure & Mesothelioma (Rate)": {
            "design": "Cohort", "outcome_type": "Rate (person-time)", "exposure_type": "Binary (2 groups)",
            "exposed_label": "Asbestos-exposed workers", "unexposed_label": "Unexposed controls",
            "cases1": 32, "py1": 8500, "cases2": 4, "py2": 9200,
            "description": (
                "**Scenario:** A retrospective cohort of shipyard workers followed using employment records. "
                "Person-time varies because workers joined and left at different times. "
                "*Adapted from Selikoff et al. (1980).*"
            )
        },
    }

    col_title, col_reset = st.columns([5, 1])
    with col_title:
        st.markdown("#### 💡 Load a Preset Scenario")
        st.caption("Choose a preset to pre-fill all fields with realistic data, or select \'I\'ll enter my own data\' to start from scratch.")
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_tab1", help="Clear all fields and return to defaults"):
            keys_to_clear = [
                "preset_choice", "last_preset", "design", "outcome_type", "exposure_type",
                "row_0", "row_1", "col_0", "col_1",
                "cell_0_0", "cell_0_1", "cell_1_0", "cell_1_1",
                "exposed_label", "unexposed_label", "cases1", "cases2", "py1", "py2"
            ]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # Track the last loaded preset so we only inject session state once per change
    if "last_preset" not in st.session_state:
        st.session_state["last_preset"] = None

    preset_choice = st.selectbox("Select a scenario:", list(PRESETS.keys()), key="preset_choice")
    preset = PRESETS[preset_choice]

    # When preset changes, write all values into session state so widgets pick them up
    if preset_choice != st.session_state["last_preset"]:
        st.session_state["last_preset"] = preset_choice
        if preset:
            for key in ["design", "outcome_type", "exposure_type",
                        "row_0", "row_1", "col_0", "col_1",
                        "cell_0_0", "cell_0_1", "cell_1_0", "cell_1_1",
                        "exposed_label", "unexposed_label",
                        "cases1", "py1", "cases2", "py2"]:
                if key in preset:
                    st.session_state[key] = preset[key]
        else:
            # Reset to defaults
            st.session_state["row_0"] = "Group 1"
            st.session_state["row_1"] = "Group 2"
            st.session_state["col_0"] = "Level 1"
            st.session_state["col_1"] = "Level 2"
            st.session_state["cell_0_0"] = 0
            st.session_state["cell_0_1"] = 0
            st.session_state["cell_1_0"] = 0
            st.session_state["cell_1_1"] = 0
            st.session_state["cases1"] = 0
            st.session_state["cases2"] = 0
            st.session_state["py1"] = 1
            st.session_state["py2"] = 1
            st.session_state["exposed_label"] = "Exposed"
            st.session_state["unexposed_label"] = "Unexposed"
            st.session_state["design"] = "Cohort"
            st.session_state["outcome_type"] = "Binary"
            st.session_state["exposure_type"] = "Binary (2 groups)"
        st.rerun()

    if preset:
        st.info(preset["description"])

    st.divider()

    def pval(key, default):
        return st.session_state.get(key, preset[key] if preset and key in preset else default)

    st.subheader("Step 1️⃣: Study Design")

    design_options = ["Cohort", "Case-Control", "Cross-sectional"]
    if "design" not in st.session_state:
        st.session_state["design"] = "Cohort"
    design = st.selectbox(
        "Select study design:",
        design_options,
        index=design_options.index(st.session_state.get("design", "Cohort")),
        help=(
            "Cohort: Follow exposed and unexposed groups forward in time to compare new cases (incidence). "
            "Produces a Risk Ratio (RR) or Rate Ratio.\n\n"
            "Case-Control: Start with people who already have the disease (cases) and those who don't (controls), "
            "then look back at their past exposure. Produces an Odds Ratio (OR).\n\n"
            "Cross-sectional: Measure exposure and outcome at the same point in time — a snapshot. "
            "Produces a Prevalence Ratio (PR) or Odds Ratio."
        )
    )

    st.subheader("Step 2️⃣: Outcome Variable Type")

    outcome_options = ["Binary", "Categorical (Nominal >2 levels)", "Ordinal", "Rate (person-time)"]
    if "outcome_type" not in st.session_state:
        st.session_state["outcome_type"] = "Binary"
    outcome_type = st.selectbox(
        "Select outcome type:",
        outcome_options,
        index=outcome_options.index(st.session_state.get("outcome_type", "Binary")),
        help=(
            "Binary: The outcome has exactly two categories (e.g., disease: Yes/No). "
            "Produces a 2x2 table and allows RR and OR calculation.\n\n"
            "Categorical (Nominal >2 levels): The outcome has 3+ unordered categories "
            "(e.g., disease severity: mild, moderate, severe). Produces a larger table; chi-square only.\n\n"
            "Ordinal: The outcome has ordered categories (e.g., none, mild, severe). "
            "Treated like categorical here; chi-square only.\n\n"
            "Rate (person-time): Each participant contributes a different amount of time at risk. "
            "Used when follow-up time varies. Produces an Incidence Rate Ratio (IRR)."
        )
    )

    st.subheader("Step 3️⃣: Exposure Variable Type")

    exposure_options = ["Binary (2 groups)", "Categorical (>2 groups)"]
    if "exposure_type" not in st.session_state:
        st.session_state["exposure_type"] = "Binary (2 groups)"
    exposure_type = st.selectbox(
        "Select exposure type:",
        exposure_options,
        index=exposure_options.index(st.session_state.get("exposure_type", "Binary (2 groups)")),
        help=(
            "Binary (2 groups): Exposed vs. unexposed — the most common setup. "
            "Enables a standard 2x2 table and full RR/OR analysis.\n\n"
            "Categorical (>2 groups): Three or more exposure levels "
            "(e.g., never smoker, light smoker, heavy smoker). "
            "Produces a larger table; chi-square only."
        )
    )

    st.divider()

    # ----------------------------------------------------------
    # RATE OUTCOME
    # ----------------------------------------------------------

    if outcome_type == "Rate (person-time)":

        st.header("📊 Rate Data Entry")
        st.info(
            "**Person-time** accounts for the fact that participants are followed for different lengths of time. "
            "Instead of counting people, you sum up the total time each person was at risk. "
            "The **Incidence Rate Ratio (IRR)** compares the rate of new cases between the two groups."
        )

        exposed_label = st.text_input("Label for Exposed Group", key="exposed_label")
        unexposed_label = st.text_input("Label for Unexposed Group", key="unexposed_label")

        col1, col2 = st.columns(2)
        with col1:
            cases1 = st.number_input(f"Cases ({exposed_label})", min_value=0, key="cases1")
            py1 = st.number_input(f"Person-Time ({exposed_label})", min_value=1, key="py1")
        with col2:
            cases2 = st.number_input(f"Cases ({unexposed_label})", min_value=0, key="cases2")
            py2 = st.number_input(f"Person-Time ({unexposed_label})", min_value=1, key="py2")

        if st.button("Run Statistical Analysis"):
            if cases1 > 0 and cases2 > 0:
                ir1 = cases1 / py1
                ir2 = cases2 / py2
                if ir2 > 0:
                    rr = ir1 / ir2
                    se_log_rr = math.sqrt((1/cases1) + (1/cases2))
                    ci_low = math.exp(math.log(rr) - 1.96 * se_log_rr)
                    ci_high = math.exp(math.log(rr) + 1.96 * se_log_rr)

                    st.subheader("📈 Rate Ratio Results")
                    st.write(f"Rate Ratio = {round(rr,3)}")
                    st.write(f"95% CI: ({round(ci_low,3)}, {round(ci_high,3)})")

                    if ci_low <= 1 <= ci_high:
                        st.warning(
                            f"**Result:** The incidence rate among {exposed_label} does not differ "
                            f"statistically from {unexposed_label} (IRR = {round(rr,2)}, "
                            f"95% CI: {round(ci_low,2)}–{round(ci_high,2)}). "
                            f"Because the confidence interval includes 1, we **fail to reject** the null "
                            f"hypothesis of no association."
                        )
                    else:
                        direction = "higher" if rr > 1 else "lower"
                        st.success(
                            f"**Result:** In this {design.lower()} study, the incidence rate among "
                            f"{exposed_label} is {round(rr,2)} times {direction} than "
                            f"among {unexposed_label} (IRR = {round(rr,2)}, "
                            f"95% CI: {round(ci_low,2)}–{round(ci_high,2)}). "
                            f"Because the CI does not include 1, we **reject the null hypothesis**."
                        )
                    draw_ci("IRR", rr, ci_low, ci_high)
            else:
                st.warning("Both groups must have at least one case to compute rate ratio.")

    # ----------------------------------------------------------
    # CONTINGENCY TABLE
    # ----------------------------------------------------------

    elif outcome_type in ["Binary", "Categorical (Nominal >2 levels)", "Ordinal"]:

        st.header("📊 Build Contingency Table")

        if exposure_type == "Binary (2 groups)":
            num_rows = 2
        else:
            num_rows = st.number_input("Number of Exposure Groups", min_value=2, value=3)

        if outcome_type == "Binary":
            num_cols = 2
        else:
            num_cols = st.number_input("Number of Outcome Levels", min_value=2, value=3)

        st.subheader("Label Exposure Groups")
        st.caption(
            "Name each exposure group — these are the groups you are comparing. "
            "Examples: *Received medication* vs *No medication*; *Smoker* vs *Non-smoker*; "
            "*Vaccinated* vs *Unvaccinated*; *Exposed to toxin* vs *Not exposed*."
        )
        row_names = [
            st.text_input(
                f"Exposure Group {i+1}", key=f"row_{i}",
                help="Examples: Exposed, Unexposed, Vaccinated, Smoker, Received Treatment, Control group."
            )
            for i in range(num_rows)
        ]

        st.subheader("Label Outcome Levels")
        st.caption(
            "Name each outcome category — this is what you measured or counted. "
            "Examples: *Hypertension* vs *No hypertension*; *Asthma exacerbation* vs *No exacerbation*; "
            "*Lung cancer* vs *No lung cancer*; *Hospitalized* vs *Not hospitalized*."
        )
        col_names = [
            st.text_input(
                f"Outcome Level {j+1}", key=f"col_{j}",
                help="Examples: Disease, No disease, Outcome present, Outcome absent, Mild, Moderate, Severe."
            )
            for j in range(num_cols)
        ]

        st.subheader("Enter Cell Counts")

        data = []
        for i in range(num_rows):
            st.markdown(f"**{row_names[i]}**")
            row = []
            cols = st.columns(num_cols)
            for j in range(num_cols):
                with cols[j]:
                    value = st.number_input(
                        f"{row_names[i]} - {col_names[j]}",
                        min_value=0,
                        key=f"cell_{i}_{j}"
                    )
                    row.append(value)
            data.append(row)

        df = pd.DataFrame(data, columns=col_names, index=row_names)
        df["Row Total"] = df.sum(axis=1)
        total_row = df.sum()
        total_row.name = "Column Total"
        df = pd.concat([df, total_row.to_frame().T])

        st.subheader("Contingency Table with Totals")
        st.table(df)

        table_array = df.iloc[:-1, :-1].values

        if st.button("Run Statistical Analysis"):

            alpha = 0.05

            if np.sum(table_array) == 0:
                st.warning(
                    "⚠️ All cell counts are zero. Please enter your data before running the analysis. "
                    "Tip: Label your groups first, then fill in the counts in each cell."
                )
            else:
                row_sums = table_array.sum(axis=1)
                col_sums = table_array.sum(axis=0)

                if np.any(row_sums == 0) or np.any(col_sums == 0):
                    st.warning(
                        "⚠️ One or more rows or columns sum to zero. "
                        "Each exposure group and outcome category must have at least one observation "
                        "before the chi-square test can run."
                    )
                else:
                    try:
                        chi2, p, dof, expected = chi2_contingency(table_array)
                    except ValueError:
                        st.warning("Chi-square could not be computed due to structural zeros.")
                    else:
                        st.subheader("📈 Chi-Square Test of Independence")
                        st.info(
                            "The **chi-square (χ²) test** asks: is there a statistically significant association "
                            "between exposure and outcome, or could the pattern in the table be due to chance? "
                            "A **p-value < 0.05** means we reject the null hypothesis of no association. "
                            "A **p-value ≥ 0.05** means we do not have enough evidence to conclude an association exists."
                        )

                        st.write(f"χ²({dof}) = {round(chi2, 3)}")
                        if p < 0.0001:
                            st.write("p-value < 0.0001")
                        elif p < 0.001:
                            st.write(f"p-value = {round(p, 5)}")
                        else:
                            st.write(f"p-value = {round(p, 4)}")

                        exposure_label_string = ", ".join(row_names)
                        outcome_label_string = ", ".join(col_names)

                        if p < alpha:
                            st.success(
                                f"**Result:** In this {design.lower()} study, the distribution of "
                                f"{outcome_label_string} differs significantly across exposure groups "
                                f"({exposure_label_string}) (χ²({dof}) = {round(chi2,3)}, p = {round(p,4)}). "
                                f"We **reject the null hypothesis** of independence."
                            )
                        else:
                            st.warning(
                                f"**Result:** In this {design.lower()} study, there is insufficient "
                                f"evidence to conclude that {outcome_label_string} differs across "
                                f"exposure groups ({exposure_label_string}) "
                                f"(χ²({dof}) = {round(chi2,3)}, p = {round(p,4)}). "
                                f"We **fail to reject the null hypothesis**."
                            )

                        if num_rows == 2 and num_cols == 2:

                            a, b = table_array[0]
                            c, d = table_array[1]

                            if all(v > 0 for v in [a, b, c, d]):

                                st.info(
                                    "Because you have a 2×2 table, we can also calculate the **Risk Ratio (RR)** "
                                    "and **Odds Ratio (OR)**. The **95% confidence interval (CI)** gives the range of "
                                    "plausible values for the true measure. "
                                    "If the CI includes 1, the association is **not statistically significant** — "
                                    "an RR or OR of 1 means no difference between groups."
                                )

                                rr = (a/(a+b)) / (c/(c+d))
                                se_log_rr = math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+d)))
                                ci_low_rr = math.exp(math.log(rr)-1.96*se_log_rr)
                                ci_high_rr = math.exp(math.log(rr)+1.96*se_log_rr)

                                st.subheader("Risk Ratio (RR)")
                                st.caption(
                                    "RR = risk in exposed ÷ risk in unexposed. "
                                    "RR = 1: no difference. RR > 1: higher risk in exposed group. RR < 1: lower risk in exposed group. "
                                    "Most appropriate for cohort studies."
                                )

                                if ci_low_rr <= 1 <= ci_high_rr:
                                    st.warning(
                                        f"RR = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). "
                                        f"The CI includes 1 → **not statistically significant**. "
                                        f"We cannot conclude that {row_names[0]} has a different risk of {col_names[0]} than {row_names[1]}."
                                    )
                                else:
                                    direction = "higher" if rr > 1 else "lower"
                                    st.success(
                                        f"RR = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). "
                                        f"The risk of {col_names[0]} among {row_names[0]} is {round(rr,2)} times "
                                        f"{direction} than among {row_names[1]}. "
                                        f"The CI does not include 1 → **statistically significant**."
                                    )

                                draw_ci("RR", rr, ci_low_rr, ci_high_rr)

                                or_val = (a*d)/(b*c)
                                se_log_or = math.sqrt(1/a+1/b+1/c+1/d)
                                ci_low_or = math.exp(math.log(or_val)-1.96*se_log_or)
                                ci_high_or = math.exp(math.log(or_val)+1.96*se_log_or)

                                st.subheader("Odds Ratio (OR)")
                                st.caption(
                                    "OR = odds of outcome in exposed ÷ odds of outcome in unexposed. "
                                    "OR = 1: no difference. OR > 1: higher odds in exposed group. OR < 1: lower odds in exposed group. "
                                    "Most appropriate for case-control studies. Note: the OR is always farther from 1 than the RR."
                                )

                                if ci_low_or <= 1 <= ci_high_or:
                                    st.warning(
                                        f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). "
                                        f"The CI includes 1 → **not statistically significant**. "
                                        f"We cannot conclude that the odds of {col_names[0]} differ between {row_names[0]} and {row_names[1]}."
                                    )
                                else:
                                    direction = "higher" if or_val > 1 else "lower"
                                    st.success(
                                        f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). "
                                        f"The odds of {col_names[0]} among {row_names[0]} are {round(or_val,2)} times "
                                        f"{direction} than among {row_names[1]}. "
                                        f"The CI does not include 1 → **statistically significant**."
                                    )

                                draw_ci("OR", or_val, ci_low_or, ci_high_or)

                            else:
                                st.info(
                                    "⚠️ RR and OR require all four cells (a, b, c, d) to be greater than zero. "
                                    "At least one cell is currently 0, so these measures cannot be calculated."
                                )

    st.markdown("---")
    st.markdown("Strong epidemiologists think structurally before computing.")

# ==========================================================
# TAB 3: STANDARDIZATION
# ==========================================================

with tab3:

    col_title3, col_reset3 = st.columns([5, 1])
    with col_reset3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_tab3", help="Clear all fields and return to defaults"):
            keys_to_clear3 = ["std_preset_choice"]
            for k in keys_to_clear3:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    st.markdown("""
    **Standardization** allows fair comparison of rates between populations that differ in age structure.
    Without it, a population with more elderly people will always appear sicker — even if age-specific
    rates are identical to a younger population.

    There are two methods:
    - **Direct standardization** applies each population's age-specific rates to a single standard population structure.
    - **Indirect standardization** applies a reference population's rates to your study population, producing an **SMR**.
    """)

    st.divider()

    # ----------------------------------------------------------
    # PRESET SCENARIOS
    # ----------------------------------------------------------

    STD_PRESETS = {
        "None — I'll enter my own data": None,
        "Urban vs. Rural CVD Mortality": {
            "description": (
                "**Scenario:** Compare cardiovascular disease (CVD) mortality between an urban county "
                "(younger population) and a rural county (older population). Without standardization, "
                "the rural county appears to have higher CVD mortality — but is this real or just "
                "because it has more elderly residents? *Adapted from CDC WONDER mortality data.*"
            ),
            "age_groups": ["0–44", "45–54", "55–64", "65–74", "75+"],
            "std_pop":    [150000, 40000, 35000, 25000, 15000],
            "pop_a":      [80000,  15000, 12000,  8000,  4000],
            "deaths_a":   [12,     45,    120,    280,   310],
            "pop_b":      [30000,  18000, 22000,  20000, 14000],
            "deaths_b":   [5,      55,    145,    430,   580],
            "label_a": "Urban County",
            "label_b": "Rural County",
            "outcome": "CVD deaths",
            "ref_label": "State population"
        },
        "Miners vs. Office Workers (Lung Disease)": {
            "description": (
                "**Scenario:** Compare lung disease mortality between coal miners and office workers. "
                "Miners are older on average due to longer tenure requirements. "
                "Standardization reveals whether excess mortality is due to age or occupational exposure. "
                "*Adapted from NIOSH occupational cohort data.*"
            ),
            "age_groups": ["20–34", "35–44", "45–54", "55–64", "65–74"],
            "std_pop":    [5000,    6000,    5500,    4000,    2000],
            "pop_a":      [800,     1800,    2100,    1600,    900],
            "deaths_a":   [1,       6,       18,      38,      32],
            "pop_b":      [2000,    2200,    1800,    1200,    600],
            "deaths_b":   [0,       2,       5,       10,      8],
            "label_a": "Coal Miners",
            "label_b": "Office Workers",
            "outcome": "lung disease deaths",
            "ref_label": "Workforce population"
        },
        "Country A vs. Country B (Breast Cancer)": {
            "description": (
                "**Scenario:** Compare breast cancer mortality between two countries with very different "
                "age distributions. Country A has a younger population (developing nation); "
                "Country B has an older population (developed nation). "
                "*Adapted from WHO Global Cancer Observatory data.*"
            ),
            "age_groups": ["0–34", "35–44", "45–54", "55–64", "65+"],
            "std_pop":    [100000, 30000,  25000,  20000,  25000],
            "pop_a":      [60000,  12000,   8000,   5000,   3000],
            "deaths_a":   [2,      18,      35,     42,     38],
            "pop_b":      [20000,  18000,  22000,  20000,  25000],
            "deaths_b":   [1,      22,      58,     95,    142],
            "label_a": "Country A",
            "label_b": "Country B",
            "outcome": "breast cancer deaths",
            "ref_label": "World standard population"
        },
    }

    std_preset_choice = st.selectbox(
        "Load a preset scenario:",
        list(STD_PRESETS.keys()),
        key="std_preset_choice"
    )
    std_preset = STD_PRESETS[std_preset_choice]

    if std_preset:
        st.info(std_preset["description"])

    st.divider()

    # ----------------------------------------------------------
    # DATA ENTRY
    # ----------------------------------------------------------

    if std_preset:
        age_groups  = std_preset["age_groups"]
        std_pop     = std_preset["std_pop"]
        pop_a       = std_preset["pop_a"]
        deaths_a    = std_preset["deaths_a"]
        pop_b       = std_preset["pop_b"]
        deaths_b    = std_preset["deaths_b"]
        label_a     = std_preset["label_a"]
        label_b     = std_preset["label_b"]
        outcome_lbl = std_preset["outcome"]
        ref_label   = std_preset["ref_label"]
        n_groups    = len(age_groups)
    else:
        st.subheader("Set Up Your Data")

        col1, col2 = st.columns(2)
        with col1:
            label_a = st.text_input("Population A name", "Population A")
            label_b = st.text_input("Population B name", "Population B")
        with col2:
            ref_label   = st.text_input("Standard/reference population name", "Standard Population")
            outcome_lbl = st.text_input("Outcome label", "deaths")

        n_groups = st.number_input("Number of age groups", min_value=2, max_value=10, value=5)
        age_groups, std_pop, pop_a, deaths_a, pop_b, deaths_b = [], [], [], [], [], []

        st.markdown("Enter data for each age group:")
        header_cols = st.columns([2, 2, 2, 2, 2, 2])
        header_cols[0].markdown("**Age Group**")
        header_cols[1].markdown(f"**{ref_label} size**")
        header_cols[2].markdown(f"**{label_a} pop.**")
        header_cols[3].markdown(f"**{label_a} {outcome_lbl}**")
        header_cols[4].markdown(f"**{label_b} pop.**")
        header_cols[5].markdown(f"**{label_b} {outcome_lbl}**")

        for i in range(int(n_groups)):
            cols = st.columns([2, 2, 2, 2, 2, 2])
            age_groups.append(cols[0].text_input("", f"Group {i+1}", key=f"ag_{i}", label_visibility="collapsed"))
            std_pop.append(cols[1].number_input("", min_value=1, value=10000, key=f"sp_{i}", label_visibility="collapsed"))
            pop_a.append(cols[2].number_input("", min_value=1, value=1000, key=f"pa_{i}", label_visibility="collapsed"))
            deaths_a.append(cols[3].number_input("", min_value=0, value=0, key=f"da_{i}", label_visibility="collapsed"))
            pop_b.append(cols[4].number_input("", min_value=1, value=1000, key=f"pb_{i}", label_visibility="collapsed"))
            deaths_b.append(cols[5].number_input("", min_value=0, value=0, key=f"db_{i}", label_visibility="collapsed"))

    # ----------------------------------------------------------
    # SHOW DATA TABLE
    # ----------------------------------------------------------


    if st.button("Run Standardization Analysis"):

      if sum(pop_a) == 0 or sum(pop_b) == 0 or sum(std_pop) == 0:
        st.warning("⚠️ Population sizes cannot be zero. Please check your data.")
      else:

        rate_a = [deaths_a[i] / max(pop_a[i], 1) * 100000 for i in range(n_groups)]
        rate_b = [deaths_b[i] / max(pop_b[i], 1) * 100000 for i in range(n_groups)]
        ref_rate = [(deaths_a[i] + deaths_b[i]) / max(pop_a[i] + pop_b[i], 1) * 100000 for i in range(n_groups)]

        display_df = pd.DataFrame({
            "Age Group": age_groups,
            f"{ref_label} (std pop)": std_pop,
            f"{label_a} — Population": pop_a,
            f"{label_a} — {outcome_lbl.capitalize()}": deaths_a,
            f"{label_a} — Rate per 100k": [round(r, 1) for r in rate_a],
            f"{label_b} — Population": pop_b,
            f"{label_b} — {outcome_lbl.capitalize()}": deaths_b,
            f"{label_b} — Rate per 100k": [round(r, 1) for r in rate_b],
        })

        with st.expander("📋 View age-stratified data table", expanded=True):
            st.dataframe(display_df, use_container_width=True)

        st.divider()

        # ----------------------------------------------------------
        # CALCULATIONS
        # ----------------------------------------------------------

        # DIRECT STANDARDIZATION
        # Apply each population's age-specific rates to the standard population
        expected_a_direct = [rate_a[i] / 100000 * std_pop[i] for i in range(n_groups)]
        expected_b_direct = [rate_b[i] / 100000 * std_pop[i] for i in range(n_groups)]
        total_std_pop = sum(std_pop)
        age_adj_rate_a = sum(expected_a_direct) / total_std_pop * 100000
        age_adj_rate_b = sum(expected_b_direct) / total_std_pop * 100000

        # INDIRECT STANDARDIZATION (SMR)
        # Apply reference rates to each population's age structure
        expected_a_indirect = [ref_rate[i] / 100000 * pop_a[i] for i in range(n_groups)]
        expected_b_indirect = [ref_rate[i] / 100000 * pop_b[i] for i in range(n_groups)]
        total_obs_a = sum(deaths_a)
        total_obs_b = sum(deaths_b)
        total_exp_a = sum(expected_a_indirect)
        total_exp_b = sum(expected_b_indirect)
        smr_a = total_obs_a / total_exp_a if total_exp_a > 0 else None
        smr_b = total_obs_b / total_exp_b if total_exp_b > 0 else None

        # Crude rates (unadjusted)
        crude_rate_a = sum(deaths_a) / sum(pop_a) * 100000
        crude_rate_b = sum(deaths_b) / sum(pop_b) * 100000

        # ----------------------------------------------------------
        # RESULTS: SIDE BY SIDE
        # ----------------------------------------------------------

        st.subheader("📊 Results: Side-by-Side Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {label_a}")
            st.metric("Crude Rate (per 100,000)", round(crude_rate_a, 1))
            st.metric("Age-Adjusted Rate — Direct (per 100,000)", round(age_adj_rate_a, 1))
            if smr_a:
                st.metric("SMR — Indirect", round(smr_a, 3))

        with col2:
            st.markdown(f"### {label_b}")
            st.metric("Crude Rate (per 100,000)", round(crude_rate_b, 1))
            st.metric("Age-Adjusted Rate — Direct (per 100,000)", round(age_adj_rate_b, 1))
            if smr_b:
                st.metric("SMR — Indirect", round(smr_b, 3))

        st.divider()

        # ----------------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------------

        st.subheader("🔍 Interpretation")

        # Crude comparison
        crude_higher = label_a if crude_rate_a > crude_rate_b else label_b
        crude_lower  = label_b if crude_rate_a > crude_rate_b else label_a
        crude_diff   = abs(crude_rate_a - crude_rate_b)

        # Direct comparison
        adj_higher = label_a if age_adj_rate_a > age_adj_rate_b else label_b
        adj_lower  = label_b if age_adj_rate_a > age_adj_rate_b else label_a
        adj_diff   = abs(age_adj_rate_a - age_adj_rate_b)

        st.markdown(f"""
    **Crude rates (unadjusted):**
    {label_a} has a crude rate of {round(crude_rate_a,1)} per 100,000; {label_b} has {round(crude_rate_b,1)} per 100,000.
    {crude_higher} appears to have higher {outcome_lbl} by {round(crude_diff,1)} per 100,000 before accounting for age.
        """)

        st.markdown(f"""
    **After direct standardization (age-adjusted rates):**
    Applying both populations' rates to the same standard population ({ref_label}):
    {label_a} age-adjusted rate = {round(age_adj_rate_a,1)} per 100,000;
    {label_b} age-adjusted rate = {round(age_adj_rate_b,1)} per 100,000.
    {adj_higher} has a higher age-adjusted rate by {round(adj_diff,1)} per 100,000.
        """)

        if crude_higher != adj_higher:
            st.error(
                f"⚠️ **Confounding by age detected!** The crude rates suggested {crude_higher} had "
                f"higher {outcome_lbl}, but after age adjustment, {adj_higher} actually has the higher rate. "
                f"This reversal indicates that age was confounding the crude comparison — "
                f"the apparent difference was largely due to differences in age structure, not true disease burden."
            )
        else:
            diff_pct = abs(crude_rate_a - crude_rate_b) / max(crude_rate_a, crude_rate_b, 0.0001) * 100
            adj_pct  = abs(age_adj_rate_a - age_adj_rate_b) / max(age_adj_rate_a, age_adj_rate_b, 0.0001) * 100
            if abs(diff_pct - adj_pct) > 10:
                st.warning(
                    f"⚠️ Age partially confounded this comparison. The gap between populations "
                    f"changed after adjustment, suggesting age structure was inflating or deflating the crude difference. "
                    f"The age-adjusted rates give a fairer comparison."
                )
            else:
                st.success(
                    f"✅ Age structure had minimal impact here. The crude and age-adjusted rates tell "
                    f"a similar story, suggesting age is not a major confounder in this comparison."
                )

        if smr_a and smr_b:
            st.markdown(f"""
    **Indirect standardization (SMR):**
    - {label_a} SMR = {round(smr_a,3)}: observed {outcome_lbl} were {round(smr_a,3)}x the number expected based on reference rates.
    {"  → Excess mortality compared to reference population." if smr_a > 1 else "  → Lower mortality than reference population (possible healthy worker effect)."}
    - {label_b} SMR = {round(smr_b,3)}: observed {outcome_lbl} were {round(smr_b,3)}x the number expected.
    {"  → Excess mortality compared to reference population." if smr_b > 1 else "  → Lower mortality than reference population."}
            """)

        st.divider()

        # ----------------------------------------------------------
        # EXPLAINER
        # ----------------------------------------------------------

        with st.expander("📖 When to use direct vs. indirect standardization"):
            st.markdown("""
    **Direct standardization:**
    - Apply each study population's age-specific rates to one shared standard population
    - Produces an **age-adjusted rate** (comparable across populations)
    - Requires knowing age-specific rates in both populations
    - Best when you want to compare two or more populations fairly
    - Result depends on choice of standard population (WHO world standard, US 2000 standard, etc.)

    **Indirect standardization (SMR):**
    - Apply a reference population's age-specific rates to your study population's age structure
    - Produces an **SMR** (ratio of observed to expected events)
    - Use when your study population is small and age-specific rates are unstable
    - Best for comparing one group against a well-established reference (e.g., national rates)
    - Watch for the **healthy worker effect**: workers are often healthier than the general population,
      producing SMR < 1 even without a true protective exposure

    **Key difference:**
    | | Direct | Indirect |
    |---|---|---|
    | What you apply | Study pop's rates → standard pop structure | Reference rates → study pop structure |
    | Output | Age-adjusted rate (per 100,000) | SMR (ratio) |
    | Best use | Comparing multiple populations | Comparing one group vs. a reference |
    | Requires | Age-specific rates in study populations | Reference population age-specific rates |
    | Sensitive to | Choice of standard population | Size/stability of reference rates |
            """)

        st.markdown("---")
        st.markdown("Strong epidemiologists think structurally before computing.")


    # TAB 2: ADVANCED EPI MEASURES
    # ==========================================================

    with tab2:

        col_title2, col_reset2 = st.columns([5, 1])
        with col_title2:
            st.markdown(
                "Calculate advanced epidemiologic measures using **preset realistic scenarios** "
                "or enter your own data manually."
            )
        with col_reset2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Reset", key="reset_tab2", help="Clear all fields and return to defaults"):
                keys_to_clear2 = [
                    "smr_mode", "smr_scenario", "ar_mode", "ar_scenario",
                    "nnt_mode", "nnt_scenario", "hr_mode", "hr_scenario"
                ]
                for k in keys_to_clear2:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        measure = st.selectbox(
            "Select measure to calculate:",
            [
                "Population Attributable Risk (PAR)",
                "Standardized Mortality Ratio (SMR)",
                "Attributable Risk & AR%",
                "Number Needed to Harm / Treat (NNH/NNT)",
                "Hazard Ratio (HR)"
            ]
        )

        st.divider()

        # ----------------------------------------------------------
        # POPULATION ATTRIBUTABLE RISK (PAR)
        # ----------------------------------------------------------

        if measure == "Population Attributable Risk (PAR)":

            st.subheader("Population Attributable Risk (PAR)")
            st.info(
                "**PAR** estimates the proportion of disease in the **total population** that is attributable to "
                "a specific exposure — in other words, how much disease could theoretically be prevented "
                "if the exposure were eliminated. It requires knowing both the **Risk Ratio** and the "
                "**prevalence of the exposure in the population**."
            )

            with st.expander("📖 Formula"):
                st.markdown("""
                **PAR% = Pe × (RR − 1) / [1 + Pe × (RR − 1)] × 100**

                Where:
                - **Pe** = prevalence of exposure in the population
                - **RR** = Risk Ratio (relative risk)

                Also known as the **Population Attributable Fraction (PAF)**.
                """)

            data_mode = st.radio("Data entry mode", ["Use preset scenario", "Enter my own data"], horizontal=True)

            if data_mode == "Use preset scenario":
                scenario = st.selectbox("Choose a scenario", [
                    "Smoking & Lung Cancer (US Adults)",
                    "Physical Inactivity & Type 2 Diabetes",
                    "Obesity & Cardiovascular Disease"
                ])

                if scenario == "Smoking & Lung Cancer (US Adults)":
                    Pe, RR = 0.14, 15.0
                    st.markdown("""
                    **Scenario:** Approximately 14% of U.S. adults currently smoke cigarettes.
                    Smokers have roughly 15 times the risk of developing lung cancer compared to non-smokers.
                    *Sources: CDC (2023), IARC Monographs.*
                    """)
                elif scenario == "Physical Inactivity & Type 2 Diabetes":
                    Pe, RR = 0.46, 1.5
                    st.markdown("""
                    **Scenario:** About 46% of U.S. adults do not meet physical activity guidelines.
                    Physically inactive individuals have approximately 1.5 times the risk of developing
                    Type 2 diabetes compared to those who are active.
                    *Sources: CDC BRFSS (2022), Jeon et al., Diabetes Care.*
                    """)
                elif scenario == "Obesity & Cardiovascular Disease":
                    Pe, RR = 0.42, 2.0
                    st.markdown("""
                    **Scenario:** Approximately 42% of U.S. adults have obesity (BMI ≥ 30).
                    Individuals with obesity have about twice the risk of cardiovascular disease
                    compared to those with healthy weight.
                    *Sources: CDC NHANES (2022), American Heart Association.*
                    """)

            else:
                Pe = st.number_input(
                    "Prevalence of exposure in the population (Pe)",
                    min_value=0.001, max_value=0.999, value=0.30, step=0.01,
                    help="Enter as a proportion between 0 and 1. Example: 0.25 means 25% of the population is exposed."
                )
                RR = st.number_input(
                    "Risk Ratio (RR)",
                    min_value=0.01, value=2.0, step=0.1,
                    help="The relative risk of disease in the exposed group compared to the unexposed group."
                )

            if st.button("Calculate PAR"):
                PAR_pct = (Pe * (RR - 1)) / (1 + Pe * (RR - 1)) * 100

                st.subheader("📈 Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Exposure Prevalence (Pe)", f"{round(Pe*100,1)}%")
                col2.metric("Risk Ratio (RR)", round(RR, 2))
                col3.metric("PAR%", f"{round(PAR_pct, 1)}%")

                st.success(
                    f"**Interpretation:** {round(PAR_pct,1)}% of cases in the total population are attributable "
                    f"to this exposure. If the exposure were completely eliminated, we would expect to prevent "
                    f"approximately {round(PAR_pct,1)}% of all cases of this disease in the population. "
                    f"This assumes the association is causal."
                )

                if PAR_pct > 50:
                    st.warning("⚠️ A PAR% above 50% suggests this exposure is a dominant driver of disease burden in the population.")

        # ----------------------------------------------------------
        # STANDARDIZED MORTALITY RATIO (SMR)
        # ----------------------------------------------------------

        elif measure == "Standardized Mortality Ratio (SMR)":

            st.subheader("Standardized Mortality Ratio (SMR)")
            st.info(
                "The **SMR** compares the observed number of deaths (or cases) in a study population "
                "to the number that would be **expected** based on the rates of a reference (standard) population. "
                "It is used to assess whether a specific group — such as workers in a particular industry — "
                "experiences more or fewer deaths than the general population."
            )

            with st.expander("📖 Formula"):
                st.markdown("""
                **SMR = Observed Deaths / Expected Deaths**

                - **SMR = 1.0**: The study group has the same mortality as the reference population
                - **SMR > 1.0**: Higher mortality than expected (excess deaths)
                - **SMR < 1.0**: Lower mortality than expected (healthy worker effect)

                **Expected deaths** are calculated by applying the reference population's age-specific
                death rates to the age distribution of the study population.
                """)

            data_mode = st.radio("Data entry mode", ["Use preset scenario", "Enter my own data"], horizontal=True, key="smr_mode")

            if data_mode == "Use preset scenario":
                scenario = st.selectbox("Choose a scenario", [
                    "Coal Miners & Respiratory Disease",
                    "Nuclear Plant Workers & All-Cause Mortality",
                    "Firefighters & Cancer Mortality"
                ], key="smr_scenario")

                if scenario == "Coal Miners & Respiratory Disease":
                    age_groups = ["20–34", "35–44", "45–54", "55–64", "65–74"]
                    observed =  [2,  8, 22, 41, 35]
                    ref_rates = [0.0003, 0.0010, 0.0038, 0.0092, 0.0198]
                    pop_sizes = [1200, 1800, 2100, 1600,  900]
                    st.markdown("""
                    **Scenario:** A cohort of 7,600 underground coal miners is followed for 10 years.
                    Deaths from respiratory disease are compared to age-specific rates in the general
                    male working population. *Adapted from NIOSH occupational cohort studies.*
                    """)

                elif scenario == "Nuclear Plant Workers & All-Cause Mortality":
                    age_groups = ["20–34", "35–44", "45–54", "55–64", "65–74"]
                    observed =  [3, 10, 18, 29, 22]
                    ref_rates = [0.0008, 0.0018, 0.0045, 0.0110, 0.0240]
                    pop_sizes = [2000, 2500, 1800, 1200,  600]
                    st.markdown("""
                    **Scenario:** A cohort of 8,100 nuclear power plant workers is followed for 10 years.
                    All-cause mortality is compared to age-specific rates in the general population.
                    This scenario often demonstrates the **healthy worker effect**.
                    *Adapted from published nuclear worker cohort studies.*
                    """)

                elif scenario == "Firefighters & Cancer Mortality":
                    age_groups = ["20–34", "35–44", "45–54", "55–64", "65–74"]
                    observed =  [1, 6, 19, 38, 31]
                    ref_rates = [0.0001, 0.0006, 0.0024, 0.0068, 0.0160]
                    pop_sizes = [1500, 2000, 1900, 1400,  800]
                    st.markdown("""
                    **Scenario:** A cohort of 7,600 career firefighters is followed for 10 years.
                    Cancer mortality is compared to age-specific rates in the general male population.
                    *Adapted from Daniels et al. (2014) and IAFF mortality studies.*
                    """)

                st.divider()
                st.markdown("**Age-Stratified Data:**")
                smr_df = pd.DataFrame({
                    "Age Group": age_groups,
                    "Study Population Size": pop_sizes,
                    "Observed Deaths": observed,
                    "Reference Rate (per person)": ref_rates,
                    "Expected Deaths": [round(pop_sizes[i] * ref_rates[i], 2) for i in range(len(age_groups))]
                })
                st.table(smr_df)

                total_observed = sum(observed)
                total_expected = sum([pop_sizes[i] * ref_rates[i] for i in range(len(age_groups))])

            else:
                st.markdown("Enter observed and expected deaths by age group:")
                n_groups = st.number_input("Number of age groups", min_value=1, max_value=10, value=3)
                observed = []
                expected_list = []
                for i in range(n_groups):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        ag = st.text_input(f"Age group {i+1} label", f"Group {i+1}", key=f"smr_ag_{i}")
                    with c2:
                        obs = st.number_input(f"Observed deaths", min_value=0, key=f"smr_obs_{i}")
                    with c3:
                        exp = st.number_input(f"Expected deaths", min_value=0.0, step=0.1, key=f"smr_exp_{i}")
                    observed.append(obs)
                    expected_list.append(exp)
                total_observed = sum(observed)
                total_expected = sum(expected_list)

            if st.button("Calculate SMR"):
                if total_expected > 0:
                    smr = total_observed / total_expected

                    # 95% CI using Poisson approximation
                    ci_low_smr = smr - 1.96 * (smr / math.sqrt(total_observed)) if total_observed > 0 else 0
                    ci_high_smr = smr + 1.96 * (smr / math.sqrt(total_observed)) if total_observed > 0 else 0
                    ci_low_smr = max(0, ci_low_smr)

                    st.subheader("📈 Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Observed Deaths", int(total_observed))
                    col2.metric("Expected Deaths", round(total_expected, 2))
                    col3.metric("SMR", round(smr, 3))

                    st.write(f"95% CI: ({round(ci_low_smr,3)}, {round(ci_high_smr,3)})")

                    if ci_low_smr <= 1 <= ci_high_smr:
                        st.warning(
                            f"**Interpretation:** SMR = {round(smr,2)} (95% CI: {round(ci_low_smr,2)}–{round(ci_high_smr,2)}). "
                            f"The confidence interval includes 1.0, so we cannot conclude that mortality in this "
                            f"population differs significantly from the reference population."
                        )
                    elif smr > 1:
                        st.error(
                            f"**Interpretation:** SMR = {round(smr,2)} (95% CI: {round(ci_low_smr,2)}–{round(ci_high_smr,2)}). "
                            f"There were {int(total_observed)} observed deaths vs. {round(total_expected,1)} expected. "
                            f"Mortality in this population is {round(smr,2)} times higher than in the reference population — "
                            f"a statistically significant excess. We reject the null hypothesis (SMR = 1)."
                        )
                    else:
                        st.success(
                            f"**Interpretation:** SMR = {round(smr,2)} (95% CI: {round(ci_low_smr,2)}–{round(ci_high_smr,2)}). "
                            f"There were {int(total_observed)} observed deaths vs. {round(total_expected,1)} expected. "
                            f"Mortality in this population is lower than in the reference population — "
                            f"this may reflect the **healthy worker effect** (workers are generally healthier than the general population)."
                        )

                    draw_ci("SMR", smr, ci_low_smr, ci_high_smr)

        # ----------------------------------------------------------
        # ATTRIBUTABLE RISK & AR%
        # ----------------------------------------------------------

        elif measure == "Attributable Risk & AR%":

            st.subheader("Attributable Risk (AR) & Attributable Risk Percent (AR%)")
            st.info(
                "**Attributable Risk (AR)**, also called the **Risk Difference**, measures the absolute "
                "difference in risk between exposed and unexposed groups. "
                "**AR%** expresses that difference as a percentage of the exposed group's total risk — "
                "it asks: of all the disease in the exposed group, what fraction is due to the exposure?"
            )

            with st.expander("📖 Formulas"):
                st.markdown("""
                **AR (Risk Difference) = Risk in Exposed − Risk in Unexposed**

                **AR% = (AR / Risk in Exposed) × 100**

                Or equivalently: **AR% = (RR − 1) / RR × 100**

                - AR tells you the **absolute** excess risk due to exposure
                - AR% tells you the **proportion** of exposed group's risk attributable to exposure
                - Contrast with RR, which is a *relative* measure
                """)

            data_mode = st.radio("Data entry mode", ["Use preset scenario", "Enter my own data"], horizontal=True, key="ar_mode")

            if data_mode == "Use preset scenario":
                scenario = st.selectbox("Choose a scenario", [
                    "Hypertension & Cardiovascular Disease",
                    "Unvaccinated Children & Measles",
                    "High Sodium Diet & Stroke"
                ], key="ar_scenario")

                if scenario == "Hypertension & Cardiovascular Disease":
                    r_exposed, r_unexposed = 0.12, 0.04
                    st.markdown("""
                    **Scenario:** In a 10-year cohort study, adults with hypertension had a 12% risk
                    of cardiovascular disease (CVD), compared to 4% among normotensive adults.
                    *Adapted from Framingham Heart Study estimates.*
                    """)
                elif scenario == "Unvaccinated Children & Measles":
                    r_exposed, r_unexposed = 0.90, 0.02
                    st.markdown("""
                    **Scenario:** During a measles outbreak in an unvaccinated community, 90% of
                    unvaccinated children developed measles, compared to 2% of vaccinated children.
                    *Adapted from CDC outbreak investigation data.*
                    """)
                elif scenario == "High Sodium Diet & Stroke":
                    r_exposed, r_unexposed = 0.08, 0.03
                    st.markdown("""
                    **Scenario:** In a prospective cohort, adults consuming >5g sodium/day had an 8%
                    10-year stroke risk, compared to 3% among those consuming <2g/day.
                    *Adapted from He & MacGregor, J Human Hypertension (2003).*
                    """)

            else:
                r_exposed = st.number_input(
                    "Risk in exposed group (as proportion)",
                    min_value=0.001, max_value=1.0, value=0.12, step=0.01,
                    help="Example: 0.12 = 12% of exposed individuals developed the outcome."
                )
                r_unexposed = st.number_input(
                    "Risk in unexposed group (as proportion)",
                    min_value=0.001, max_value=1.0, value=0.04, step=0.01,
                    help="Example: 0.04 = 4% of unexposed individuals developed the outcome."
                )

            if st.button("Calculate AR & AR%"):
                ar = r_exposed - r_unexposed
                rr = r_exposed / r_unexposed
                ar_pct = (ar / r_exposed) * 100

                st.subheader("📈 Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Risk (Exposed)", f"{round(r_exposed*100,1)}%")
                col2.metric("Risk (Unexposed)", f"{round(r_unexposed*100,1)}%")
                col3.metric("AR (Risk Difference)", f"{round(ar*100,1)}%")
                col4.metric("AR%", f"{round(ar_pct,1)}%")

                st.success(
                    f"**AR Interpretation:** The exposed group had {round(ar*100,1)} additional cases per 100 people "
                    f"compared to the unexposed group. This is the absolute excess risk due to exposure."
                )
                st.success(
                    f"**AR% Interpretation:** Of all disease occurring in the exposed group, "
                    f"{round(ar_pct,1)}% is attributable to the exposure. "
                    f"If the exposure were removed, we could theoretically prevent {round(ar_pct,1)}% "
                    f"of cases among exposed individuals."
                )
                st.info(
                    f"**Note:** The Risk Ratio (RR) for this data is {round(rr,2)}. "
                    f"While the RR tells you the *relative* increase in risk, "
                    f"the AR tells you the *absolute* excess — which is often more meaningful for public health decision-making."
                )

        # ----------------------------------------------------------
        # NNH / NNT
        # ----------------------------------------------------------

        elif measure == "Number Needed to Harm / Treat (NNH/NNT)":

            st.subheader("Number Needed to Harm (NNH) / Number Needed to Treat (NNT)")
            st.info(
                "**NNT** is the number of people who need to receive a treatment for one additional person "
                "to benefit. **NNH** is the number of people who need to be exposed to a risk factor for "
                "one additional person to be harmed. Both are the inverse of the Attributable Risk (Risk Difference) "
                "and express risk in a clinically intuitive way."
            )

            with st.expander("📖 Formulas"):
                st.markdown("""
                **NNT = 1 / (Risk in control − Risk in treated)**  *(used when treatment reduces risk)*

                **NNH = 1 / (Risk in exposed − Risk in unexposed)**  *(used when exposure increases risk)*

                Both are the inverse of the **Attributable Risk (Risk Difference)**.

                - A smaller NNT means the treatment is more effective
                - A smaller NNH means the exposure is more dangerous
                """)

            data_mode = st.radio("Data entry mode", ["Use preset scenario", "Enter my own data"], horizontal=True, key="nnt_mode")

            if data_mode == "Use preset scenario":
                scenario = st.selectbox("Choose a scenario", [
                    "Statins & Major Cardiac Events (NNT)",
                    "Aspirin & GI Bleeding (NNH)",
                    "Smoking Cessation Program & Quitting at 1 Year (NNT)"
                ], key="nnt_scenario")

                if scenario == "Statins & Major Cardiac Events (NNT)":
                    r_treatment, r_control = 0.04, 0.06
                    label_treatment, label_control = "Statin therapy", "Placebo"
                    context = "NNT — Treatment reduces risk"
                    st.markdown("""
                    **Scenario:** In a 5-year RCT, 4% of patients on statin therapy experienced a major
                    cardiac event, compared to 6% in the placebo group.
                    *Adapted from Heart Protection Study (MRC/BHF).*
                    """)
                elif scenario == "Aspirin & GI Bleeding (NNH)":
                    r_treatment, r_control = 0.025, 0.010
                    label_treatment, label_control = "Daily aspirin", "No aspirin"
                    context = "NNH — Exposure increases risk"
                    st.markdown("""
                    **Scenario:** In a cohort study, 2.5% of adults taking daily aspirin experienced
                    a GI bleeding event over 3 years, compared to 1.0% among non-users.
                    *Adapted from US Preventive Services Task Force aspirin evidence review.*
                    """)
                elif scenario == "Smoking Cessation Program & Quitting at 1 Year (NNT)":
                    r_treatment, r_control = 0.22, 0.08
                    label_treatment, label_control = "Cessation program", "No program"
                    context = "NNT — Treatment increases benefit (quitting)"
                    st.markdown("""
                    **Scenario:** In a randomized trial, 22% of participants in a structured smoking
                    cessation program successfully quit at 1 year, vs. 8% in the control group.
                    *Adapted from Cochrane systematic review on behavioral smoking cessation.*
                    """)

            else:
                label_treatment = st.text_input("Label for treatment/exposed group", "Treatment")
                label_control = st.text_input("Label for control/unexposed group", "Control")
                r_treatment = st.number_input(
                    f"Risk in {label_treatment} group",
                    min_value=0.001, max_value=1.0, value=0.04, step=0.01
                )
                r_control = st.number_input(
                    f"Risk in {label_control} group",
                    min_value=0.001, max_value=1.0, value=0.06, step=0.01
                )
                context = "Manual entry"

            if st.button("Calculate NNH / NNT"):
                risk_diff = abs(r_treatment - r_control)
                nnt_nnh = round(1 / risk_diff, 1) if risk_diff > 0 else None

                st.subheader("📈 Results")
                col1, col2, col3 = st.columns(3)
                col1.metric(f"Risk ({label_treatment})", f"{round(r_treatment*100,1)}%")
                col2.metric(f"Risk ({label_control})", f"{round(r_control*100,1)}%")
                col3.metric("Risk Difference (AR)", f"{round(risk_diff*100,1)}%")

                if nnt_nnh:
                    if r_treatment < r_control:
                        st.success(
                            f"**NNT = {nnt_nnh}**. You would need to treat **{nnt_nnh} people** with "
                            f"{label_treatment} to prevent one additional outcome compared to {label_control}. "
                            f"A lower NNT means a more effective intervention."
                        )
                    else:
                        st.error(
                            f"**NNH = {nnt_nnh}**. For every **{nnt_nnh} people** exposed to "
                            f"{label_treatment}, one additional harm would be expected compared to {label_control}. "
                            f"A lower NNH means a more dangerous exposure."
                        )

                    # ---- NNT/NNH Interpretation Guide ----
                    st.subheader("📐 How to Interpret This Number")

                    if r_treatment < r_control:
                        # NNT guidance
                        st.markdown("""
    **Is this NNT good or bad?** Context is everything. An NNT that seems large can still represent
    an important public health benefit — especially for serious outcomes like heart attacks or death,
    or when a treatment is low-cost and low-risk.

    **General benchmarks for NNT:**
    | NNT Range | General Interpretation |
    |-----------|----------------------|
    | 1 – 5 | Highly effective. Nearly everyone treated benefits (e.g., antibiotics for strep throat). |
    | 6 – 15 | Very effective. Strong benefit for a meaningful proportion of patients. |
    | 16 – 50 | Moderately effective. Common for preventive interventions in general populations. |
    | 51 – 100 | Modest effect. May still be worthwhile if the outcome is severe or treatment is cheap. |
    | > 100 | Small effect per person. Meaningful only at population scale or for catastrophic outcomes. |

    **Real-world NNT examples for comparison:**
    | Intervention | NNT | Outcome | Timeframe |
    |---|---|---|---|
    | Tamiflu for influenza | ~14 | Reduce duration by 1 day | Per illness |
    | Statins (high-risk patients) | ~20 | Prevent 1 MI or stroke | 5 years |
    | Statins (low-risk / primary prevention) | ~50–100 | Prevent 1 cardiac event | 5 years |
    | Aspirin for secondary MI prevention | ~40 | Prevent 1 death or MI | 2 years |
    | Smoking cessation counseling | ~10–20 | 1 additional person quits | 1 year |
    | Seatbelt use | ~3,300 | Prevent 1 death per crash | Per crash |

    **Key principle:** A large NNT is not automatically bad. Ask:
    - How serious is the outcome? (Preventing 1 death in 200 patients may be very worthwhile.)
    - What are the costs and risks of treatment? (Low side effects = higher acceptable NNT.)
    - What is the baseline risk? (Low-risk populations always produce higher NNTs for preventive interventions.)
                        """)
                    else:
                        # NNH guidance
                        st.markdown("""
    **Is this NNH concerning?** Context determines whether an NNH represents an acceptable risk or a serious safety signal.

    **General benchmarks for NNH:**
    | NNH Range | General Interpretation |
    |-----------|----------------------|
    | 1 – 10 | Very high harm rate. Exposure is extremely dangerous for this outcome. |
    | 11 – 50 | High harm rate. Serious safety concern requiring careful risk-benefit analysis. |
    | 51 – 200 | Moderate harm rate. Common in drug side effect studies; must be weighed against benefits. |
    | 201 – 1,000 | Low harm rate. May be acceptable depending on severity of harm and magnitude of benefit. |
    | > 1,000 | Very rare harm. Usually acceptable unless the outcome is catastrophic (e.g., death). |

    **Real-world NNH examples for comparison:**
    | Exposure / Drug | NNH | Harm | Timeframe |
    |---|---|---|---|
    | Daily aspirin (low-dose) | ~67 | GI bleeding event | 3 years |
    | NSAIDs (regular use) | ~100 | GI complications | 1 year |
    | COX-2 inhibitors (Vioxx) | ~150 | Cardiovascular event | 18 months |
    | Smoking (1+ pack/day) | ~7 | Lung cancer | Lifetime |
    | Unvaccinated (measles outbreak) | ~1.1 | Measles infection | Per outbreak |

    **Key principle:** NNH must always be interpreted alongside NNT (the benefit).
    - If NNT < NNH: more people benefit than are harmed — generally favorable.
    - If NNH < NNT: more people are harmed than benefit — raises serious safety concerns.
    - Always consider: How severe is the harm vs. the benefit?
                        """)

        # ----------------------------------------------------------
        # HAZARD RATIO
        # ----------------------------------------------------------

        elif measure == "Hazard Ratio (HR)":

            st.subheader("Hazard Ratio (HR)")
            st.info(
                "The **Hazard Ratio** compares the rate at which an event occurs over time between two groups. "
                "Unlike the Risk Ratio, which looks at cumulative risk over a fixed period, the HR accounts for "
                "**when** events happen — making it the appropriate measure for survival analysis and "
                "time-to-event studies. It is the standard measure reported in Cox proportional hazards regression."
            )

            with st.expander("📖 Key concepts"):
                st.markdown("""
                **HR = Hazard in exposed group / Hazard in unexposed group**

                - **HR = 1.0**: Events occur at the same rate in both groups
                - **HR > 1.0**: Events occur faster in the exposed group (increased hazard)
                - **HR < 1.0**: Events occur more slowly in the exposed group (protective)

                **How it differs from RR:**
                - RR compares cumulative risk at a fixed time point
                - HR compares the *instantaneous rate* of events at any given moment
                - HR is more appropriate when follow-up time varies across participants
                - HR is the standard output of Cox proportional hazards models

                **When to use survival analysis:**
                - Participants enter and leave the study at different times
                - Some participants are lost to follow-up (censored)
                - You are interested in *time to event*, not just whether it occurred
                """)

            data_mode = st.radio("Data entry mode", ["Use preset scenario", "Enter my own data"], horizontal=True, key="hr_mode")

            if data_mode == "Use preset scenario":
                scenario = st.selectbox("Choose a scenario", [
                    "Statins & Time to First MI (RCT)",
                    "HIV Diagnosis & Time to AIDS (Cohort)",
                    "Physical Activity & Time to Dementia Onset"
                ], key="hr_scenario")

                if scenario == "Statins & Time to First MI (RCT)":
                    hr, ci_low_hr, ci_high_hr = 0.68, 0.54, 0.85
                    exposed_label, unexposed_label = "Statin therapy", "Placebo"
                    outcome_label = "first myocardial infarction"
                    st.markdown("""
                    **Scenario:** In a 5-year randomized controlled trial, participants were assigned to
                    statin therapy or placebo and followed for time to first myocardial infarction (MI).
                    Results are reported from a Cox proportional hazards model.
                    *Adapted from JUPITER trial (Ridker et al., NEJM 2008).*
                    """)
                elif scenario == "HIV Diagnosis & Time to AIDS (Cohort)":
                    hr, ci_low_hr, ci_high_hr = 2.31, 1.74, 3.07
                    exposed_label, unexposed_label = "CD4 < 200 at diagnosis", "CD4 ≥ 200 at diagnosis"
                    outcome_label = "AIDS-defining illness"
                    st.markdown("""
                    **Scenario:** A prospective cohort of HIV-positive adults is followed for time to
                    AIDS-defining illness. Participants are stratified by CD4 count at HIV diagnosis.
                    *Adapted from published HIV natural history cohort data.*
                    """)
                elif scenario == "Physical Activity & Time to Dementia Onset":
                    hr, ci_low_hr, ci_high_hr = 0.72, 0.58, 0.89
                    exposed_label, unexposed_label = "High physical activity", "Low physical activity"
                    outcome_label = "dementia diagnosis"
                    st.markdown("""
                    **Scenario:** Adults aged 65+ are followed for up to 10 years. Those with high
                    physical activity levels are compared to sedentary adults for time to dementia diagnosis.
                    *Adapted from Larson et al., Annals of Internal Medicine (2006).*
                    """)

                st.subheader("📈 Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Hazard Ratio (HR)", round(hr, 2))
                col2.metric("95% CI Lower", round(ci_low_hr, 2))
                col3.metric("95% CI Upper", round(ci_high_hr, 2))

                if ci_low_hr <= 1 <= ci_high_hr:
                    st.warning(
                        f"**Interpretation:** HR = {round(hr,2)} (95% CI: {round(ci_low_hr,2)}–{round(ci_high_hr,2)}). "
                        f"The CI includes 1 → **not statistically significant**. "
                        f"We cannot conclude that the rate of {outcome_label} differs between {exposed_label} and {unexposed_label}."
                    )
                elif hr < 1:
                    st.success(
                        f"**Interpretation:** HR = {round(hr,2)} (95% CI: {round(ci_low_hr,2)}–{round(ci_high_hr,2)}). "
                        f"At any given point in time, {exposed_label} had {round((1-hr)*100,1)}% lower hazard of {outcome_label} "
                        f"compared to {unexposed_label}. The CI does not include 1 → **statistically significant**."
                    )
                else:
                    st.error(
                        f"**Interpretation:** HR = {round(hr,2)} (95% CI: {round(ci_low_hr,2)}–{round(ci_high_hr,2)}). "
                        f"At any given point in time, {exposed_label} had {round((hr-1)*100,1)}% higher hazard of {outcome_label} "
                        f"compared to {unexposed_label}. The CI does not include 1 → **statistically significant**."
                    )

                draw_ci("HR", hr, ci_low_hr, ci_high_hr)

            else:
                hr = st.number_input("Hazard Ratio (HR)", min_value=0.01, value=0.68, step=0.01)
                ci_low_hr = st.number_input("95% CI Lower bound", min_value=0.001, value=0.54, step=0.01)
                ci_high_hr = st.number_input("95% CI Upper bound", min_value=0.001, value=0.85, step=0.01)
                exposed_label = st.text_input("Exposed group label", "Exposed")
                unexposed_label = st.text_input("Unexposed group label", "Unexposed")
                outcome_label = st.text_input("Outcome label", "the outcome")

                if st.button("Interpret HR"):
                    st.subheader("📈 Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Hazard Ratio (HR)", round(hr, 2))
                    col2.metric("95% CI Lower", round(ci_low_hr, 2))
                    col3.metric("95% CI Upper", round(ci_high_hr, 2))

                    if ci_low_hr <= 1 <= ci_high_hr:
                        st.warning(
                            f"**Interpretation:** HR = {round(hr,2)} (95% CI: {round(ci_low_hr,2)}–{round(ci_high_hr,2)}). "
                            f"The CI includes 1 → **not statistically significant**."
                        )
                    elif hr < 1:
                        st.success(
                            f"**Interpretation:** HR = {round(hr,2)} (95% CI: {round(ci_low_hr,2)}–{round(ci_high_hr,2)}). "
                            f"{exposed_label} had {round((1-hr)*100,1)}% lower hazard of {outcome_label} "
                            f"compared to {unexposed_label}. CI does not include 1 → **statistically significant**."
                        )
                    else:
                        st.error(
                            f"**Interpretation:** HR = {round(hr,2)} (95% CI: {round(ci_low_hr,2)}–{round(ci_high_hr,2)}). "
                            f"{exposed_label} had {round((hr-1)*100,1)}% higher hazard of {outcome_label} "
                            f"compared to {unexposed_label}. CI does not include 1 → **statistically significant**."
                        )

                    draw_ci("HR", hr, ci_low_hr, ci_high_hr)

    st.markdown("---")
    st.markdown("Strong epidemiologists think structurally before computing.")

# ==========================================================
# TAB 4: PRACTICE ON YOUR OWN
# ==========================================================

with tab4:

    st.markdown("""
    Read each scenario carefully, then make your decisions about study design,
    outcome variable type, and exposure variable type. You'll get immediate feedback
    on each choice explaining why it is or isn't correct.
    """)

    PRACTICE_SCENARIOS = [
        {
            "id": "scenario_1",
            "title": "Scenario 1: Lead Exposure & Cognitive Development",
            "description": (
                "Investigators recruit 400 children aged 6–12 from neighborhoods near a former "
                "lead smelting plant and 400 children from neighborhoods with no known lead exposure. "
                "Blood lead levels are measured at baseline. Children are followed for 3 years, "
                "and new diagnoses of learning disabilities are recorded. Researchers want to compare "
                "the rate of new learning disability diagnoses between the two groups."
            ),
            "correct_design": "Cohort",
            "correct_outcome": "Binary",
            "correct_exposure": "Binary (2 groups)",
            "data": {
                "type": "contingency",
                "context": (
                    "Here is the data collected from the 3-year follow-up. "
                    "Enter these counts into the analysis to calculate the Risk Ratio, Odds Ratio, and p-value."
                ),
                "row_names": ["Lead-exposed", "Unexposed"],
                "col_names": ["Learning Disability", "No Learning Disability"],
                "cells": [[52, 348], [21, 379]],
            },
            "design_hint": (
                "Think about the timeline. Researchers identified exposed and unexposed children "
                "at the start, then followed them forward to see who developed a new diagnosis. "
                "This forward-looking approach — from exposure to outcome — is the defining feature "
                "of a cohort study. Case-control studies start with people who already have the disease."
            ),
            "outcome_hint": (
                "The outcome is a learning disability diagnosis — either a child receives one or they don't. "
                "That's a yes/no outcome, which is binary. If the outcome had multiple unordered categories "
                "(e.g., mild, moderate, severe, none) it would be categorical."
            ),
            "exposure_hint": (
                "There are two exposure groups: children near the smelting plant vs. children with no known "
                "lead exposure. Two groups = binary exposure. If the researchers had compared low, medium, "
                "and high lead neighborhoods, that would be categorical (>2 groups)."
            ),
        },
        {
            "id": "scenario_2",
            "title": "Scenario 2: Fast Food Consumption & Obesity",
            "description": (
                "A public health team conducts a one-time survey of 2,500 adults at community health fairs "
                "across three cities. At the time of the survey, each participant reports how many times "
                "per week they eat fast food, and their height and weight are measured to determine "
                "current obesity status (BMI ≥ 30 vs. BMI < 30). The team wants to examine the "
                "relationship between fast food frequency and obesity as it exists right now."
            ),
            "correct_design": "Cross-sectional",
            "correct_outcome": "Binary",
            "correct_exposure": "Categorical (>2 groups)",
            "data": {
                "type": "contingency_wide",
                "context": (
                    "Here is the survey data organized by fast food frequency and obesity status. "
                    "Run the chi-square test to see whether the association is statistically significant."
                ),
                "row_names": ["Never", "1–2x/week", "3–4x/week", "5+x/week"],
                "col_names": ["Obese", "Not Obese"],
                "cells": [
                    [62,  538],
                    [118, 682],
                    [189, 561],
                    [141, 209],
                ],
            },
            "design_hint": (
                "The key phrase is 'one-time survey' and 'as it exists right now.' Both exposure "
                "(fast food frequency) and outcome (obesity status) are measured at the same point in time — "
                "a snapshot. That's a cross-sectional study. In a cohort study, you would follow people "
                "over time to see who becomes obese. In a case-control study, you would start with obese "
                "and non-obese people and ask about past eating habits."
            ),
            "outcome_hint": (
                "Obesity status is BMI ≥ 30 vs. BMI < 30 — two categories, so it's binary. "
                "If the researchers had categorized BMI into underweight, normal, overweight, and obese, "
                "that would be categorical with more than 2 levels."
            ),
            "exposure_hint": (
                "Fast food frequency has more than two levels — for example: never, 1–2 times/week, "
                "3–4 times/week, 5+ times/week. That's categorical with more than 2 groups. "
                "If the only comparison were 'eats fast food' vs. 'does not eat fast food,' "
                "that would be binary."
            ),
        },
        {
            "id": "scenario_3",
            "title": "Scenario 3: HPV Vaccine & Cervical Cancer",
            "description": (
                "Researchers identify 250 women aged 25–45 with confirmed cervical cancer diagnoses "
                "from a hospital registry. They also recruit 500 women without cervical cancer from "
                "the same hospital system, matched on age and clinic location. Medical records and "
                "patient interviews are used to determine whether each woman received the HPV vaccine "
                "during adolescence. The researchers want to know whether unvaccinated women are more "
                "likely to develop cervical cancer."
            ),
            "correct_design": "Case-Control",
            "correct_outcome": "Binary",
            "correct_exposure": "Binary (2 groups)",
            "data": {
                "type": "contingency",
                "context": (
                    "Here is the case-control data from medical records and interviews. "
                    "Because this is a case-control study, the Odds Ratio is the appropriate measure of association."
                ),
                "row_names": ["Unvaccinated", "Vaccinated"],
                "col_names": ["Cervical Cancer (Case)", "No Cancer (Control)"],
                "cells": [[178, 182], [72, 318]],
            },
            "design_hint": (
                "The researchers started by identifying people who already have the disease (cases: cervical "
                "cancer) and people who don't (controls), then looked back at past vaccination history. "
                "Starting with outcome status and looking backward is the hallmark of a case-control study. "
                "A cohort study would have enrolled women in adolescence and followed them forward to see "
                "who developed cancer — which would take decades."
            ),
            "outcome_hint": (
                "In a case-control study, the 'outcome' is what defines your cases vs. controls — here, "
                "cervical cancer: yes or no. That's binary. The outcome was already determined before "
                "the study began, which is why you're looking backward at exposure."
            ),
            "exposure_hint": (
                "HPV vaccination status is received the vaccine vs. did not receive the vaccine — "
                "two groups, so it's binary. If the researchers had compared unvaccinated, partially "
                "vaccinated (1–2 doses), and fully vaccinated (3 doses), that would be categorical "
                "with more than 2 groups."
            ),
        },
        {
            "id": "scenario_4",
            "title": "Scenario 4: Shift Work & Metabolic Syndrome",
            "description": (
                "An occupational health study enrolls 1,200 hospital employees and classifies them "
                "into three groups based on their work schedule: day shift only, rotating shift "
                "(alternates between day and night), and permanent night shift. Employees are "
                "followed for 5 years. At the end of follow-up, researchers assess whether each "
                "employee has developed metabolic syndrome (yes/no), a cluster of conditions "
                "including high blood pressure, high blood sugar, and excess abdominal fat."
            ),
            "correct_design": "Cohort",
            "correct_outcome": "Binary",
            "correct_exposure": "Categorical (>2 groups)",
            "data": {
                "type": "contingency_wide",
                "context": (
                    "Here is the 5-year follow-up data for the 1,200 hospital employees by shift type. "
                    "Run the chi-square test to assess whether shift type is associated with metabolic syndrome."
                ),
                "row_names": ["Day shift only", "Rotating shift", "Permanent night shift"],
                "col_names": ["Metabolic Syndrome", "No Metabolic Syndrome"],
                "cells": [
                    [62,  338],
                    [98,  302],
                    [121, 279],
                ],
            },
            "design_hint": (
                "Employees were classified by exposure (shift type) at the start, then followed "
                "forward for 5 years to see who developed metabolic syndrome. Forward in time, "
                "from exposure to outcome — that's a cohort study. The fact that there are three "
                "exposure groups doesn't change the study design."
            ),
            "outcome_hint": (
                "Metabolic syndrome is either present or absent — yes or no. That's binary. "
                "Even though metabolic syndrome involves multiple components, the outcome being "
                "measured here is a single yes/no classification."
            ),
            "exposure_hint": (
                "There are three exposure groups: day shift, rotating shift, and night shift. "
                "Three groups means categorical with more than 2 levels. If the only comparison "
                "were 'shift worker' vs. 'non-shift worker,' that would be binary."
            ),
        },
        {
            "id": "scenario_5",
            "title": "Scenario 5: Air Pollution & Emergency Department Visits",
            "description": (
                "Researchers want to study the effect of fine particulate matter (PM2.5) air pollution "
                "on respiratory health in a city. They enroll 3,000 adults and monitor their daily "
                "PM2.5 exposure using neighborhood air quality sensors over a 2-year period. "
                "Because participants move, change jobs, and vary in time spent outdoors, each person "
                "contributes a different total amount of time at risk. The outcome of interest is "
                "new emergency department (ED) visits for respiratory illness."
            ),
            "correct_design": "Cohort",
            "correct_outcome": "Rate (person-time)",
            "correct_exposure": "Binary (2 groups)",
            "data": {
                "type": "rate",
                "context": (
                    "Here is the person-time data from the 2-year follow-up. Because participants "
                    "contributed different amounts of follow-up time, we use person-years rather than "
                    "simple counts. Calculate the Incidence Rate Ratio (IRR) to compare the two groups."
                ),
                "row_names": ["High PM2.5 exposure", "Low PM2.5 exposure"],
                "cases": [187, 64],
                "person_time": [4200, 5100],
            },
            "design_hint": (
                "Participants are followed forward over time from a defined exposure to see who "
                "develops new ED visits — a cohort study. The important clue here is that follow-up "
                "time varies across participants, which affects how the outcome is measured "
                "(but not the study design itself)."
            ),
            "outcome_hint": (
                "The key phrase is 'each person contributes a different total amount of time at risk.' "
                "When follow-up time varies, you can't simply count who got sick — you need to account "
                "for how long each person was observed. This requires a rate outcome using person-time "
                "(e.g., ED visits per 100 person-years). Binary outcome would be appropriate only "
                "if everyone was followed for the same fixed period."
            ),
            "exposure_hint": (
                "PM2.5 exposure is being compared as high vs. low (or exposed vs. unexposed to elevated levels) "
                "— two groups, so binary. If researchers had used three or more pollution categories "
                "(e.g., low, moderate, high, very high), that would be categorical with more than 2 groups."
            ),
        },
        {
            "id": "scenario_6",
            "title": "Scenario 6: Food Insecurity & Mental Health",
            "description": (
                "A state health department conducts a telephone survey of 5,000 randomly selected "
                "households. Each respondent is asked about current food insecurity status "
                "(food secure vs. food insecure) and completes a validated depression screening "
                "instrument (PHQ-9). Based on their score, respondents are classified as: "
                "no depression, mild depression, moderate depression, or severe depression. "
                "The survey is conducted once, and researchers want to describe the association "
                "between food insecurity and depression severity as it exists at the time of the survey."
            ),
            "correct_design": "Cross-sectional",
            "correct_outcome": "Categorical (Nominal >2 levels)",
            "correct_exposure": "Binary (2 groups)",
            "data": {
                "type": "contingency_wide",
                "context": (
                    "Here is the survey data showing depression severity by food insecurity status. "
                    "Run the chi-square test to assess whether the association is statistically significant."
                ),
                "row_names": ["Food Insecure", "Food Secure"],
                "col_names": ["No Depression", "Mild", "Moderate", "Severe"],
                "cells": [
                    [312, 284, 198, 106],
                    [2180, 980, 412, 28],
                ],
            },
            "design_hint": (
                "Both food insecurity and depression are measured at the same point in time — "
                "a single telephone survey. There is no follow-up period and no looking backward "
                "at past exposures. Measuring exposure and outcome simultaneously in a snapshot "
                "is the defining feature of a cross-sectional study."
            ),
            "outcome_hint": (
                "The PHQ-9 produces four categories: no depression, mild, moderate, and severe. "
                "That's more than two categories, so it's categorical (nominal with >2 levels). "
                "If the researchers had only classified respondents as depressed vs. not depressed, "
                "the outcome would be binary."
            ),
            "exposure_hint": (
                "Food insecurity is classified as food secure vs. food insecure — two groups, so binary. "
                "If the study had used multiple levels of food insecurity (e.g., food secure, "
                "marginally food insecure, food insecure, severely food insecure), "
                "that would be categorical with more than 2 groups."
            ),
        },
    ]

    design_options   = ["— Select —", "Cohort", "Case-Control", "Cross-sectional"]
    outcome_options  = ["— Select —", "Binary", "Categorical (Nominal >2 levels)", "Ordinal", "Rate (person-time)"]
    exposure_options = ["— Select —", "Binary (2 groups)", "Categorical (>2 groups)"]

    # Reset button
    col_hdr, col_rst = st.columns([5, 1])
    with col_rst:
        if st.button("🔄 Reset", key="reset_tab4", help="Clear all answers"):
            for sc in PRACTICE_SCENARIOS:
                for field in ["design", "outcome", "exposure"]:
                    k = f"prac_{sc['id']}_{field}"
                    if k in st.session_state:
                        del st.session_state[k]
            st.rerun()

    for sc in PRACTICE_SCENARIOS:

        st.divider()
        st.subheader(sc["title"])
        st.markdown(sc["description"])

        sid = sc["id"]

        # --- STUDY DESIGN ---
        st.markdown("**What is the study design?**")
        design_choice = st.selectbox(
            "Study design:", design_options,
            key=f"prac_{sid}_design", label_visibility="collapsed"
        )

        if design_choice != "— Select —":
            if design_choice == sc["correct_design"]:
                st.success(f"✅ Correct! This is a **{sc['correct_design']}** study. " + sc["design_hint"])
            else:
                st.error(f"❌ Not quite. Think about this: " + sc["design_hint"])

        # --- OUTCOME TYPE ---
        st.markdown("**What is the outcome variable type?**")
        outcome_choice = st.selectbox(
            "Outcome type:", outcome_options,
            key=f"prac_{sid}_outcome", label_visibility="collapsed"
        )

        if outcome_choice != "— Select —":
            if outcome_choice == sc["correct_outcome"]:
                st.success(f"✅ Correct! The outcome is **{sc['correct_outcome']}**. " + sc["outcome_hint"])
            else:
                st.error(f"❌ Not quite. Think about this: " + sc["outcome_hint"])

        # --- EXPOSURE TYPE ---
        st.markdown("**What is the exposure variable type?**")
        exposure_choice = st.selectbox(
            "Exposure type:", exposure_options,
            key=f"prac_{sid}_exposure", label_visibility="collapsed"
        )

        if exposure_choice != "— Select —":
            if exposure_choice == sc["correct_exposure"]:
                st.success(f"✅ Correct! The exposure is **{sc['correct_exposure']}**. " + sc["exposure_hint"])
            else:
                st.error(f"❌ Not quite. Think about this: " + sc["exposure_hint"])

        # Score summary for this scenario
        all_answered = all(
            st.session_state.get(f"prac_{sid}_{f}") not in [None, "— Select —"]
            for f in ["design", "outcome", "exposure"]
        )
        all_correct = (
            st.session_state.get(f"prac_{sid}_design") == sc["correct_design"] and
            st.session_state.get(f"prac_{sid}_outcome") == sc["correct_outcome"] and
            st.session_state.get(f"prac_{sid}_exposure") == sc["correct_exposure"]
        )

        if all_answered:
            correct_count = sum([
                st.session_state.get(f"prac_{sid}_design") == sc["correct_design"],
                st.session_state.get(f"prac_{sid}_outcome") == sc["correct_outcome"],
                st.session_state.get(f"prac_{sid}_exposure") == sc["correct_exposure"],
            ])
            if all_correct:
                st.info("🎯 Perfect score on this scenario — all three decisions correct!")
            else:
                st.info(f"📊 {correct_count}/3 correct on this scenario. Review the feedback above and try again.")

        # --- DATA TABLE & ANALYSIS (only shown when all correct) ---
        if all_correct and "data" in sc:
            st.markdown("---")
            st.markdown("### 📋 Now run the analysis")
            st.markdown(sc["data"]["context"])

            d = sc["data"]

            if d["type"] == "contingency":
                df_display = pd.DataFrame(
                    d["cells"],
                    columns=d["col_names"],
                    index=d["row_names"]
                )
                df_display["Row Total"] = df_display.sum(axis=1)
                total_row = df_display.sum()
                total_row.name = "Column Total"
                df_display = pd.concat([df_display, total_row.to_frame().T])
                st.table(df_display)

                if st.button("Run Statistical Analysis", key=f"run_{sid}"):
                    table = np.array(d["cells"])
                    a, b = table[0]
                    c, dd = table[1]

                    chi2_val, p_val, dof, _ = chi2_contingency(table)

                    st.subheader("Chi-Square Test")
                    st.write(f"χ²({dof}) = {round(chi2_val, 3)}")
                    if p_val < 0.0001:
                        st.write("p-value < 0.0001")
                    else:
                        st.write(f"p-value = {round(p_val, 4)}")

                    if p_val < 0.05:
                        st.success(f"The distribution of {d['col_names'][0]} differs significantly across groups. We reject the null hypothesis.")
                    else:
                        st.warning("Insufficient evidence to conclude an association exists. We fail to reject the null hypothesis.")

                    if all(v > 0 for v in [a, b, c, dd]):
                        rr = (a/(a+b)) / (c/(c+dd))
                        se_log_rr = math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+dd)))
                        ci_low_rr = math.exp(math.log(rr)-1.96*se_log_rr)
                        ci_high_rr = math.exp(math.log(rr)+1.96*se_log_rr)

                        or_val = (a*dd)/(b*c)
                        se_log_or = math.sqrt(1/a+1/b+1/c+1/dd)
                        ci_low_or = math.exp(math.log(or_val)-1.96*se_log_or)
                        ci_high_or = math.exp(math.log(or_val)+1.96*se_log_or)

                        st.subheader("Risk Ratio (RR)")
                        st.caption("Most appropriate for cohort studies. RR = risk in exposed ÷ risk in unexposed.")
                        if ci_low_rr <= 1 <= ci_high_rr:
                            st.warning(f"RR = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). CI includes 1 → not statistically significant.")
                        else:
                            direction = "higher" if rr > 1 else "lower"
                            st.success(f"RR = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). The risk of {d['col_names'][0]} among {d['row_names'][0]} is {round(rr,2)} times {direction} than among {d['row_names'][1]}.")
                        draw_ci("RR", rr, ci_low_rr, ci_high_rr)

                        st.subheader("Odds Ratio (OR)")
                        st.caption("Most appropriate for case-control studies. OR = odds of outcome in exposed ÷ odds in unexposed.")
                        if ci_low_or <= 1 <= ci_high_or:
                            st.warning(f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). CI includes 1 → not statistically significant.")
                        else:
                            direction = "higher" if or_val > 1 else "lower"
                            st.success(f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). The odds of {d['col_names'][0]} among {d['row_names'][0]} are {round(or_val,2)} times {direction} than among {d['row_names'][1]}.")
                        draw_ci("OR", or_val, ci_low_or, ci_high_or)

            elif d["type"] == "contingency_wide":
                # Categorical exposure or outcome (larger table)
                df_display = pd.DataFrame(
                    d["cells"],
                    columns=d["col_names"],
                    index=d["row_names"]
                )
                df_display["Row Total"] = df_display.sum(axis=1)
                total_row = df_display.sum()
                total_row.name = "Column Total"
                df_display = pd.concat([df_display, total_row.to_frame().T])
                st.table(df_display)

                if st.button("Run Statistical Analysis", key=f"run_{sid}"):
                    table = np.array(d["cells"])
                    chi2_val, p_val, dof, _ = chi2_contingency(table)

                    st.subheader("Chi-Square Test of Independence")
                    st.write(f"χ²({dof}) = {round(chi2_val, 3)}")
                    if p_val < 0.0001:
                        st.write("p-value < 0.0001")
                    else:
                        st.write(f"p-value = {round(p_val, 4)}")

                    if p_val < 0.05:
                        st.success(f"There is a statistically significant association between exposure and outcome (p = {round(p_val,4)}). We reject the null hypothesis of independence.")
                    else:
                        st.warning(f"Insufficient evidence to conclude an association exists (p = {round(p_val,4)}). We fail to reject the null hypothesis.")

                    st.info("Note: With more than 2 exposure or outcome categories, RR and OR cannot be calculated directly from a single 2×2 table. Chi-square is the appropriate test here.")

            elif d["type"] == "rate":
                df_display = pd.DataFrame({
                    "Group": d["row_names"],
                    "Cases": d["cases"],
                    "Person-Time": d["person_time"],
                    "Rate per 100,000": [round(d["cases"][i]/d["person_time"][i]*100000, 1) for i in range(len(d["cases"]))]
                })
                st.table(df_display)

                if st.button("Run Statistical Analysis", key=f"run_{sid}"):
                    c1, c2 = d["cases"]
                    pt1, pt2 = d["person_time"]
                    ir1 = c1 / pt1
                    ir2 = c2 / pt2
                    irr = ir1 / ir2
                    se_log_irr = math.sqrt((1/c1) + (1/c2))
                    ci_low_irr = math.exp(math.log(irr) - 1.96*se_log_irr)
                    ci_high_irr = math.exp(math.log(irr) + 1.96*se_log_irr)

                    st.subheader("Incidence Rate Ratio (IRR)")
                    st.write(f"IRR = {round(irr, 3)}")
                    st.write(f"95% CI: ({round(ci_low_irr,3)}, {round(ci_high_irr,3)})")
                    if ci_low_irr <= 1 <= ci_high_irr:
                        st.warning(f"IRR = {round(irr,2)} (95% CI: {round(ci_low_irr,2)}–{round(ci_high_irr,2)}). CI includes 1 → not statistically significant.")
                    else:
                        direction = "higher" if irr > 1 else "lower"
                        st.success(f"IRR = {round(irr,2)} (95% CI: {round(ci_low_irr,2)}–{round(ci_high_irr,2)}). The incidence rate among {d['row_names'][0]} is {round(irr,2)} times {direction} than among {d['row_names'][1]}.")
                    draw_ci("IRR", irr, ci_low_irr, ci_high_irr)

    st.divider()

    # Overall score
    total_correct = 0
    total_possible = len(PRACTICE_SCENARIOS) * 3
    for sc in PRACTICE_SCENARIOS:
        sid = sc["id"]
        total_correct += sum([
            st.session_state.get(f"prac_{sid}_design") == sc["correct_design"],
            st.session_state.get(f"prac_{sid}_outcome") == sc["correct_outcome"],
            st.session_state.get(f"prac_{sid}_exposure") == sc["correct_exposure"],
        ])

    answered = sum(
        1 for sc in PRACTICE_SCENARIOS
        for f in ["design", "outcome", "exposure"]
        if st.session_state.get(f"prac_{sc['id']}_{f}") not in [None, "— Select —"]
    )

    if answered > 0:
        st.subheader(f"📊 Overall Score: {total_correct} / {total_possible}")
        pct = round(total_correct / total_possible * 100)
        st.progress(pct / 100)
        if pct == 100:
            st.success("🏆 Perfect score! You have a strong grasp of study design and variable classification.")
        elif pct >= 75:
            st.info("Good work! Review any scenarios where you got feedback and make sure the reasoning clicks.")
        elif pct >= 50:
            st.warning("Keep going — re-read the scenarios you missed and pay attention to the timeline clues.")
        else:
            st.error("Review the core concepts of study design and variable types, then try again.")

    st.markdown("---")
    st.markdown("Strong epidemiologists think structurally before computing.")
