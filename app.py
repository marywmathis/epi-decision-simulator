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
    """Draw a number line showing the CI, point estimate, and null value (1)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    significant = not (ci_low <= 1 <= ci_high)
    color = "#2e7d32" if significant else "#e65100"

    # Determine axis range with padding
    all_vals = [ci_low, ci_high, estimate, 1.0]
    span = max(all_vals) - min(all_vals)
    pad = max(span * 0.3, 0.3)
    x_min = max(0.01, min(all_vals) - pad)
    x_max = max(all_vals) + pad

    fig, ax = plt.subplots(figsize=(8, 1.2))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    # Number line
    ax.plot([x_min, x_max], [0, 0], color="#cccccc", linewidth=1.5, zorder=1)

    # CI line
    ax.plot([ci_low, ci_high], [0, 0], color=color, linewidth=6, solid_capstyle="round", zorder=2)

    # Point estimate
    ax.scatter([estimate], [0], color=color, s=120, zorder=4)

    # Null line at 1
    ax.axvline(x=1, color="#333333", linewidth=1.5, linestyle="--", zorder=3)
    ax.text(1, 0.55, "1\n(null)", ha="center", va="bottom", fontsize=8, color="#333333")

    # Labels
    ax.text(ci_low, -0.55, f"{round(ci_low,2)}", ha="center", va="top", fontsize=8, color=color)
    ax.text(estimate, 0.55, f"{label}={round(estimate,2)}", ha="center", va="bottom", fontsize=8.5,
            color=color, fontweight="bold")
    ax.text(ci_high, -0.55, f"{round(ci_high,2)}", ha="center", va="top", fontsize=8, color=color)

    sig_text = "CI does not cross 1 → statistically significant" if significant else "CI crosses 1 → not statistically significant"
    ax.set_title(sig_text, fontsize=9, color=color, pad=4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    plt.tight_layout(pad=0.2)

    st.pyplot(fig)
    plt.close(fig)

st.title("🧭 Epidemiology Decision Simulator")
st.markdown("Study Design → Outcome Type → Exposure Type → Table → Run Analysis → Interpretation")

# ==========================================================
# STEP 1: STUDY DESIGN
# ==========================================================

st.subheader("Step 1️⃣: Study Design")

design = st.selectbox(
    "Select study design:",
    ["Cohort", "Case-Control", "Cross-sectional"],
    help=(
        "Cohort: Follow exposed and unexposed groups forward in time to compare new cases (incidence). "
        "Produces a Risk Ratio (RR) or Rate Ratio.\n\n"
        "Case-Control: Start with people who already have the disease (cases) and those who don't (controls), "
        "then look back at their past exposure. Produces an Odds Ratio (OR).\n\n"
        "Cross-sectional: Measure exposure and outcome at the same point in time — a 'snapshot.' "
        "Produces a Prevalence Ratio (PR) or Odds Ratio."
    )
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
    ],
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

# ==========================================================
# STEP 3: EXPOSURE TYPE
# ==========================================================

st.subheader("Step 3️⃣: Exposure Variable Type")

exposure_type = st.selectbox(
    "Select exposure type:",
    [
        "Binary (2 groups)",
        "Categorical (>2 groups)"
    ],
    help=(
        "Binary (2 groups): Exposed vs. unexposed — the most common setup. "
        "Enables a standard 2x2 table and full RR/OR analysis.\n\n"
        "Categorical (>2 groups): Three or more exposure levels "
        "(e.g., never smoker, light smoker, heavy smoker). "
        "Produces a larger table; chi-square only."
    )
)

st.divider()

# ==========================================================
# RATE OUTCOME SECTION
# ==========================================================

if outcome_type == "Rate (person-time)":

    st.header("📊 Rate Data Entry")

    st.info(
        "**Person-time** accounts for the fact that participants are followed for different lengths of time. "
        "Instead of counting people, you sum up the total time each person was at risk. "
        "The **Incidence Rate Ratio (IRR)** compares the rate of new cases between the two groups."
    )

    exposed_label = st.text_input("Label for Exposed Group", "Exposed")
    unexposed_label = st.text_input("Label for Unexposed Group", "Unexposed")

    col1, col2 = st.columns(2)

    with col1:
        cases1 = st.number_input(f"Cases ({exposed_label})", min_value=0)
        py1 = st.number_input(f"Person-Time ({exposed_label})", min_value=1)

    with col2:
        cases2 = st.number_input(f"Cases ({unexposed_label})", min_value=0)
        py2 = st.number_input(f"Person-Time ({unexposed_label})", min_value=1)

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

    # Labels
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

    df["Row Total"] = df.sum(axis=1)
    total_row = df.sum()
    total_row.name = "Column Total"
    df = pd.concat([df, total_row.to_frame().T])

    st.subheader("Contingency Table with Totals")
    st.table(df)

    table_array = df.iloc[:-1, :-1].values

    # ------------------------------------------------------
    # RUN ANALYSIS BUTTON
    # ------------------------------------------------------

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
                    st.warning(
                        "Chi-square could not be computed due to structural zeros."
                    )
                else:
                    st.subheader("📈 Chi-Square Test of Independence")

                    st.info(
                        "The **chi-square (χ²) test** asks: is there a statistically significant association "
                        "between exposure and outcome, or could the pattern in the table be due to chance? "
                        "A **p-value < 0.05** means we reject the null hypothesis of no association. "
                        "A **p-value ≥ 0.05** means we do not have enough evidence to conclude an association exists."
                    )

                    st.write(f"χ²({dof}) = {round(chi2, 3)}")

                    # Show more precision for very small p-values
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

                    # 2x2 Measures
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

                            # Risk Ratio
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
                                    f"We cannot conclude that {row_names[0]} has a different risk than {row_names[1]}."
                                )
                            else:
                                direction = "higher" if rr > 1 else "lower"
                                st.success(
                                    f"RR = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). "
                                    f"The risk of {col_names[0]} among {row_names[0]} "
                                    f"is {round(rr,2)} times {direction} than among {row_names[1]}. "
                                    f"The CI does not include 1 → **statistically significant**."
                                )

                            draw_ci("RR", rr, ci_low_rr, ci_high_rr)

                            # Odds Ratio
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
                                    f"We cannot conclude that the odds differ between {row_names[0]} and {row_names[1]}."
                                )
                            else:
                                direction = "higher" if or_val > 1 else "lower"
                                st.success(
                                    f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). "
                                    f"The odds of {col_names[0]} among {row_names[0]} "
                                    f"are {round(or_val,2)} times {direction} than among {row_names[1]}. "
                                    f"The CI does not include 1 → **statistically significant**."
                                )

                            draw_ci("OR", or_val, ci_low_or, ci_high_or)

                        else:
                            st.info(
                                "⚠️ RR and OR require all four cells (a, b, c, d) to be greater than zero. "
                                "At least one cell is currently 0, so these measures cannot be calculated. "
                                "Check your data and make sure every cell has at least one observation."
                            )

st.markdown("---")
st.markdown("Strong epidemiologists think structurally before computing.")
