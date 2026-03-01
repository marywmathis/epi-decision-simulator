import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import math

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

st.title("🧭 Epidemiology Decision Simulator")
st.markdown("Study Design → Outcome Type → Exposure Type → Table → Inference → Interpretation")

# ==========================================================
# STEP 1: STUDY DESIGN
# ==========================================================

st.subheader("Step 1️⃣: Study Design")

design = st.selectbox(
    "Select study design:",
    ["Cohort", "Case-Control", "Cross-sectional"]
)

# ==========================================================
# STEP 2: OUTCOME TYPE
# ==========================================================

st.subheader("Step 2️⃣: Outcome Variable Type")

outcome_type = st.selectbox(
    "Select outcome type:",
    [
        "Binary",
        "Categorical (Nominal >2 levels)",
        "Ordinal",
        "Rate (person-time)"
    ]
)

# ==========================================================
# STEP 3: EXPOSURE TYPE
# ==========================================================

st.subheader("Step 3️⃣: Exposure Variable Type")

exposure_type = st.selectbox(
    "Select exposure type:",
    [
        "Binary (2 groups)",
        "Categorical (>2 groups)"
    ]
)

st.divider()

# ==========================================================
# RATE OUTCOME SECTION
# ==========================================================

if outcome_type == "Rate (person-time)":

    st.header("📊 Rate Data Entry")

    exposed_label = st.text_input("Label for Exposed Group", "Exposed")
    unexposed_label = st.text_input("Label for Unexposed Group", "Unexposed")

    col1, col2 = st.columns(2)

    with col1:
        cases1 = st.number_input(f"Cases ({exposed_label})", min_value=0)
        py1 = st.number_input(f"Person-Time ({exposed_label})", min_value=1)

    with col2:
        cases2 = st.number_input(f"Cases ({unexposed_label})", min_value=0)
        py2 = st.number_input(f"Person-Time ({unexposed_label})", min_value=1)

    if py1 > 0 and py2 > 0 and cases1 > 0 and cases2 > 0:

        ir1 = cases1 / py1
        ir2 = cases2 / py2

        if ir2 > 0:

            rr = ir1 / ir2

            se_log_rr = math.sqrt((1/cases1) + (1/cases2))
            ci_low = math.exp(math.log(rr) - 1.96 * se_log_rr)
            ci_high = math.exp(math.log(rr) + 1.96 * se_log_rr)

            st.subheader("📈 Rate Ratio Results")

            st.write(f"Rate Ratio ({exposed_label} vs {unexposed_label}) = {round(rr,3)}")
            st.write(f"95% CI: ({round(ci_low,3)}, {round(ci_high,3)})")

            if ci_low <= 1 <= ci_high:
                st.warning(
                    f"In this {design.lower()} study, the incidence rate among "
                    f"{exposed_label} does not differ statistically from "
                    f"{unexposed_label} at α = 0.05 because the 95% confidence interval includes 1."
                )
            else:
                direction = "higher" if rr > 1 else "lower"
                st.success(
                    f"In this {design.lower()} study, the incidence rate among "
                    f"{exposed_label} is {round(rr,2)} times {direction} than "
                    f"among {unexposed_label} (95% CI: {round(ci_low,2)}–{round(ci_high,2)}), "
                    f"indicating statistical significance at α = 0.05."
                )

# ==========================================================
# CONTINGENCY TABLE SECTION
# ==========================================================

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

    # Labeling
    st.subheader("Label Exposure Groups")
    row_names = [
        st.text_input(f"Exposure Group {i+1}", f"Group {i+1}", key=f"row_{i}")
        for i in range(num_rows)
    ]

    st.subheader("Label Outcome Levels")
    col_names = [
        st.text_input(f"Outcome Level {j+1}", f"Level {j+1}", key=f"col_{j}")
        for j in range(num_cols)
    ]

    # Data Entry
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

    # Add totals
    df["Row Total"] = df.sum(axis=1)
    total_row = df.sum()
    total_row.name = "Column Total"
    df = pd.concat([df, total_row.to_frame().T])

    st.subheader("Contingency Table with Totals")
    st.table(df)

    table_array = df.iloc[:-1, :-1].values

    # ------------------------------------------------------
    # SAFE INFERENCE BLOCK
    # ------------------------------------------------------

    alpha = 0.05

    if np.sum(table_array) == 0:
        st.info("Enter data in the table to compute statistical tests.")

    else:

        row_sums = table_array.sum(axis=1)
        col_sums = table_array.sum(axis=0)

        if np.any(row_sums == 0) or np.any(col_sums == 0):
            st.info(
                "Chi-square test will run once every exposure group and outcome "
                "category contains at least one observation."
            )

        else:

            chi2, p, dof, expected = chi2_contingency(table_array)

            st.subheader("📈 Chi-Square Test of Independence")

            st.write(f"χ²({dof}) = {round(chi2,3)}")
            st.write(f"p-value = {round(p,4)}")

            exposure_label_string = ", ".join(row_names)
            outcome_label_string = ", ".join(col_names)

            if p < alpha:
                st.success(
                    f"In this {design.lower()} study, the distribution of "
                    f"{outcome_label_string} differs across exposure groups "
                    f"({exposure_label_string}) "
                    f"(χ²({dof}) = {round(chi2,2)}, p = {round(p,4)}). "
                    f"At α = {alpha}, we reject the null hypothesis that exposure "
                    f"and outcome are independent."
                )

                st.markdown(
                    "This indicates that at least one exposure group has a different "
                    "probability distribution of the outcome compared to the others. "
                    "The chi-square test evaluates overall association and does not "
                    "identify which specific groups differ."
                )

            else:
                st.warning(
                    f"In this {design.lower()} study, the distribution of "
                    f"{outcome_label_string} does not differ statistically across "
                    f"exposure groups ({exposure_label_string}) "
                    f"(χ²({dof}) = {round(chi2,2)}, p = {round(p,4)}). "
                    f"At α = {alpha}, we fail to reject the null hypothesis of independence."
                )

            # 2x2 RR and OR
            if num_rows == 2 and num_cols == 2:

                a, b = table_array[0]
                c, d = table_array[1]

                row1_total = a + b
                row2_total = c + d

                if all(v > 0 for v in [a, b, c, d]):

                    risk1 = a / row1_total
                    risk2 = c / row2_total
                    rr = risk1 / risk2

                    se_log_rr = math.sqrt((1/a)-(1/row1_total)+(1/c)-(1/row2_total))
                    ci_low_rr = math.exp(math.log(rr) - 1.96 * se_log_rr)
                    ci_high_rr = math.exp(math.log(rr) + 1.96 * se_log_rr)

                    st.subheader("Risk Ratio")

                    if ci_low_rr <= 1 <= ci_high_rr:
                        st.warning(
                            f"The risk ratio comparing {row_names[0]} to {row_names[1]} "
                            f"is {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). "
                            f"The interval includes 1 and is not statistically significant."
                        )
                    else:
                        direction = "higher" if rr > 1 else "lower"
                        st.success(
                            f"The risk of {col_names[0]} among {row_names[0]} "
                            f"is {round(rr,2)} times {direction} than among {row_names[1]} "
                            f"(95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)})."
                        )

                    or_val = (a*d)/(b*c)
                    se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
                    ci_low_or = math.exp(math.log(or_val) - 1.96 * se_log_or)
                    ci_high_or = math.exp(math.log(or_val) + 1.96 * se_log_or)

                    st.subheader("Odds Ratio")

                    if ci_low_or <= 1 <= ci_high_or:
                        st.warning(
                            f"The odds ratio comparing {row_names[0]} to {row_names[1]} "
                            f"is {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). "
                            f"The interval includes 1 and is not statistically significant."
                        )
                    else:
                        direction = "higher" if or_val > 1 else "lower"
                        st.success(
                            f"The odds of {col_names[0]} among {row_names[0]} "
                            f"are {round(or_val,2)} times {direction} than among {row_names[1]} "
                            f"(95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)})."
                        )

st.markdown("---")
st.markdown("Strong epidemiologists think structurally before computing.")
