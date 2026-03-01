import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
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
# RATE OUTCOME
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

    if py1 > 0 and py2 > 0:

        ir1 = cases1 / py1
        ir2 = cases2 / py2

        if ir2 > 0:
            rr = ir1 / ir2
            st.success(f"Rate Ratio ({exposed_label} vs {unexposed_label}) = {round(rr,3)}")

            st.markdown(
                f"The incidence rate among **{exposed_label}** is "
                f"{round(rr,2)} times the rate among **{unexposed_label}**."
            )

# ==========================================================
# CONTINGENCY TABLE ENGINE
# ==========================================================

elif outcome_type in ["Binary", "Categorical (Nominal >2 levels)", "Ordinal"]:

    st.header("📊 Build Contingency Table")

    # Number of exposure groups
    if exposure_type == "Binary (2 groups)":
        num_rows = 2
    else:
        num_rows = st.number_input("Number of Exposure Groups", min_value=2, value=3)

    # Number of outcome levels
    if outcome_type == "Binary":
        num_cols = 2
    else:
        num_cols = st.number_input("Number of Outcome Levels", min_value=2, value=3)

    st.subheader("Label Exposure Groups")
    row_names = []
    for i in range(num_rows):
        row_names.append(
            st.text_input(f"Exposure Group {i+1}", value=f"Group {i+1}", key=f"row_{i}")
        )

    st.subheader("Label Outcome Levels")
    col_names = []
    for j in range(num_cols):
        col_names.append(
            st.text_input(f"Outcome Level {j+1}", value=f"Level {j+1}", key=f"col_{j}")
        )

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

    if np.sum(table_array) > 0:

        chi2, p, dof, expected = chi2_contingency(table_array)

        st.subheader("📈 Chi-Square Test of Independence")

        st.success(f"χ²({dof}) = {round(chi2,3)}")
        st.success(f"p-value = {round(p,4)}")

        exposure_label_string = ", ".join(row_names)
        outcome_label_string = ", ".join(col_names)

        if p < 0.05:
            st.markdown(
                f"There is a statistically significant association between "
                f"**exposure categories ({exposure_label_string})** and "
                f"**outcome categories ({outcome_label_string})** "
                f"(χ²({dof}) = {round(chi2,2)}, p = {round(p,4)})."
            )
        else:
            st.markdown(
                f"There is no statistically significant association between "
                f"**exposure categories ({exposure_label_string})** and "
                f"**outcome categories ({outcome_label_string})** "
                f"(χ²({dof}) = {round(chi2,2)}, p = {round(p,4)})."
            )

        # If 2×2, compute RR and OR
        if num_rows == 2 and num_cols == 2:

            a = table_array[0][0]
            b = table_array[0][1]
            c = table_array[1][0]
            d = table_array[1][1]

            row1_total = a + b
            row2_total = c + d

            if row1_total > 0 and row2_total > 0:

                risk1 = a / row1_total
                risk2 = c / row2_total

                if risk2 > 0:
                    rr = risk1 / risk2
                    st.success(f"Risk Ratio ({row_names[0]} vs {row_names[1]}) = {round(rr,3)}")

                    st.markdown(
                        f"The risk of **{col_names[0]}** among "
                        f"**{row_names[0]}** is {round(rr,2)} times "
                        f"the risk among **{row_names[1]}**."
                    )

                if b > 0 and c > 0:
                    or_val = (a * d) / (b * c)
                    st.success(f"Odds Ratio ({row_names[0]} vs {row_names[1]}) = {round(or_val,3)}")

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
            st.warning("This variable meets criteria for confounding. Adjustment is recommended.")
        elif c3 == "No":
            st.info("This variable is likely a mediator, not a confounder.")
        else:
            st.success("This variable does not meet criteria for confounding.")

st.markdown("---")
st.markdown("Strong epidemiologists think structurally before computing.")
