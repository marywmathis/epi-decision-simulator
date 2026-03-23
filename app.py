import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import math
import random

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

# ==================================================
# LOGIN GATE
# ==================================================

def check_credentials(username, password):
    """Check username and password against Streamlit secrets."""
    users = st.secrets.get("users", {})
    if username in users and users[username] == password:
        return True
    return False

def login_screen():
    """Render the login form."""
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🧭 Epidemiology Decision Simulator")
        st.markdown("*EpiLab Interactive — licensed access only*")
        st.divider()
        st.markdown("**Please log in to continue.**")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log In", type="primary", use_container_width=True):
            if check_credentials(username, password):
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = username
                st.rerun()
            else:
                st.error("Incorrect username or password. Please contact your instructor if you need assistance.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Access issues? Contact your course instructor.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

def draw_ci(label, estimate, ci_low, ci_high):
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
    pct_low = to_pct(ci_low)
    pct_high = to_pct(ci_high)
    pct_est = to_pct(estimate)
    pct_null = to_pct(1.0)
    html = f"""
    <div style="background:#f9f9f9; border-radius:6px; padding:16px 12px 8px 12px; margin:8px 0 16px 0;">
      <div style="position:relative; height:60px; margin: 0 20px;">
        <div style="position:absolute; top:28px; left:0; right:0; height:2px; background:#cccccc;"></div>
        <div style="position:absolute; top:24px; left:{pct_low}%; width:{pct_high - pct_low}%; height:10px; background:{color}; border-radius:5px;"></div>
        <div style="position:absolute; top:20px; left:calc({pct_est}% - 9px); width:18px; height:18px; background:{color}; border-radius:50%;"></div>
        <div style="position:absolute; top:8px; left:{pct_null}%; width:2px; height:44px; background:#333; border-left: 2px dashed #333;"></div>
        <div style="position:absolute; top:0px; left:{pct_null}%; transform:translateX(-50%); font-size:11px; color:#333; white-space:nowrap;">1 (null)</div>
        <div style="position:absolute; top:46px; left:{pct_low}%; transform:translateX(-50%); font-size:11px; color:{color};">{round(ci_low,2)}</div>
        <div style="position:absolute; top:46px; left:{pct_est}%; transform:translateX(-50%); font-size:12px; color:{color}; font-weight:bold; white-space:nowrap;">{label} = {round(estimate,2)}</div>
        <div style="position:absolute; top:46px; left:{pct_high}%; transform:translateX(-50%); font-size:11px; color:{color};">{round(ci_high,2)}</div>
      </div>
      <div style="text-align:center; font-size:12px; color:{color}; font-style:italic; margin-top:28px;">{sig_text}</div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)

def chi2_explanation_expander(chi2_val, p_val, dof, table_array, col_names, row_names, tail_note=""):
    """Renders a 'Show me the math — Chi-Square' expander with contextual explanation."""
    from scipy.stats import chi2_contingency
    _, _, _, expected = chi2_contingency(table_array)
    num_rows, num_cols = table_array.shape

    # Find cell with largest contribution
    contributions = []
    for i in range(num_rows):
        for j in range(num_cols):
            o = table_array[i][j]
            e = expected[i][j]
            if e > 0:
                contrib = (o - e)**2 / e
                contributions.append((contrib, i, j, o, e))
    contributions.sort(reverse=True)
    top_contrib, top_i, top_j, top_o, top_e = contributions[0]
    top_cell = f"{row_names[top_i]} / {col_names[top_j]}"
    top_direction = "more" if top_o > top_e else "fewer"

    # Interpret χ² magnitude loosely
    if dof == 1:
        if chi2_val < 2.7: magnitude = "very small — likely consistent with chance"
        elif chi2_val < 3.84: magnitude = "moderate — approaching but not reaching significance"
        elif chi2_val < 6.6: magnitude = "meaningful — exceeds the conventional threshold (3.84)"
        elif chi2_val < 10.8: magnitude = "large — strong evidence against the null"
        else: magnitude = "very large — very strong evidence against the null"
    else:
        magnitude = "see p-value for interpretation at this df"

    p_str = f"< 0.0001" if p_val < 0.0001 else str(round(p_val, 4))

    with st.expander("🔢 Show me the math — Chi-Square"):
        st.markdown(f"""
**What is the chi-square statistic?**

The chi-square (χ²) test asks: *if there were truly no association between exposure and outcome, how often would we see a table this different from what we'd expect?*

It does this by comparing your **observed** cell counts (what you actually got) to the **expected** cell counts (what you'd predict if the null hypothesis were true — if exposure and outcome were completely independent).

**Your result: χ²({dof}) = {round(chi2_val, 3)}**

A χ² of {round(chi2_val, 3)} with {dof} degree(s) of freedom is **{magnitude}**.

The p-value of {p_str}{tail_note} means: if there were truly no association, you would see a chi-square this large or larger **{p_str} of the time** by chance alone.
        """)

        st.markdown("**Step 1: Your observed counts (O)**")
        obs_df = pd.DataFrame(table_array, columns=col_names, index=row_names)
        st.table(obs_df)

        st.markdown("**Step 2: Expected counts (E) — if exposure and outcome were independent**")
        st.caption("E = (Row Total × Column Total) ÷ Grand Total")
        exp_df = pd.DataFrame(
            [[round(expected[i][j], 2) for j in range(num_cols)] for i in range(num_rows)],
            columns=col_names, index=row_names
        )
        st.table(exp_df)

        st.markdown("**Step 3: Calculate each cell's contribution — (O − E)² ÷ E**")
        contrib_data = {}
        for j in range(num_cols):
            contrib_data[col_names[j]] = [
                round((table_array[i][j] - expected[i][j])**2 / expected[i][j], 3) if expected[i][j] > 0 else 0
                for i in range(num_rows)
            ]
        contrib_df = pd.DataFrame(contrib_data, index=row_names)
        st.table(contrib_df)

        st.markdown(f"""
**Step 4: Sum all contributions → χ²({dof}) = {round(chi2_val, 3)}**

**Largest contributor:** The cell **{top_cell}** had {int(top_o)} observed vs. {round(top_e, 1)} expected — {top_direction} cases than expected under the null. This cell drove {round(top_contrib/chi2_val*100, 0):.0f}% of the total χ² value.

**Step 5: Interpret**

p = {p_str}{tail_note} → {'We **reject** the null hypothesis. The data are inconsistent with independence.' if p_val < 0.05 else 'We **fail to reject** the null hypothesis. The data are consistent with independence.'}
        """)

def rr_or_explanation_expander(a, b, c, d, row_names, col_names, rr, or_val,
                                ci_low_rr, ci_high_rr, ci_low_or, ci_high_or,
                                is_cross_sectional=False):
    """Renders a 'Show me the math — RR & OR' expander with 2×2 cell labeling visual."""
    pabbr = "PR" if is_cross_sectional else "RR"
    plabel = "Prevalence Ratio (PR)" if is_cross_sectional else "Risk Ratio (RR)"
    risk_word = "prevalence" if is_cross_sectional else "risk"

    cell_style = "border:1px solid #aaa; padding:10px 16px; text-align:center; font-size:14px;"
    label_style = "border:1px solid #aaa; padding:10px 16px; text-align:center; font-size:12px; color:#555; background:#f5f5f5; font-weight:bold;"
    total_style = "border:1px solid #ccc; padding:10px 16px; text-align:center; font-size:13px; color:#555; background:#f0f0f0;"

    table_html = f"""
<div style="margin-bottom:16px;">
  <p style="font-weight:bold; font-size:13px; margin-bottom:8px;">2×2 Table Cell Labels</p>
  <table style="border-collapse:collapse; width:100%;">
    <tr>
      <td style="{label_style} background:#fff; border:none;"></td>
      <td style="{label_style}">{col_names[0]}</td>
      <td style="{label_style}">{col_names[1]}</td>
      <td style="{label_style}">Row Total</td>
    </tr>
    <tr>
      <td style="{label_style}">{row_names[0]}</td>
      <td style="{cell_style} background:#e8f4e8;"><span style="font-size:10px;color:#888;font-style:italic;">a = </span><span style="font-size:20px;font-weight:bold;color:#2e7d32;">{int(a)}</span></td>
      <td style="{cell_style} background:#fdecea;"><span style="font-size:10px;color:#888;font-style:italic;">b = </span><span style="font-size:20px;font-weight:bold;color:#c0392b;">{int(b)}</span></td>
      <td style="{total_style}">{int(a+b)}</td>
    </tr>
    <tr>
      <td style="{label_style}">{row_names[1]}</td>
      <td style="{cell_style} background:#fdecea;"><span style="font-size:10px;color:#888;font-style:italic;">c = </span><span style="font-size:20px;font-weight:bold;color:#c0392b;">{int(c)}</span></td>
      <td style="{cell_style} background:#e8f4e8;"><span style="font-size:10px;color:#888;font-style:italic;">d = </span><span style="font-size:20px;font-weight:bold;color:#2e7d32;">{int(d)}</span></td>
      <td style="{total_style}">{int(c+d)}</td>
    </tr>
    <tr>
      <td style="{label_style}">Col Total</td>
      <td style="{total_style}">{int(a+c)}</td>
      <td style="{total_style}">{int(b+d)}</td>
      <td style="{total_style}">{int(a+b+c+d)}</td>
    </tr>
  </table>
</div>
<div style="display:flex; gap:24px; margin-top:8px;">
  <div style="flex:1; background:#eef4fb; border-radius:6px; padding:12px 16px; font-size:13px;">
    <strong style="color:#1a4a7a;">{plabel} ({pabbr})</strong><br>
    {pabbr} = [a ÷ (a+b)] ÷ [c ÷ (c+d)]<br>
    = [{int(a)} ÷ {int(a+b)}] ÷ [{int(c)} ÷ {int(c+d)}]<br>
    = {round(a/(a+b),4)} ÷ {round(c/(c+d),4)}<br>
    = <strong style="color:#1a4a7a;">{round(rr,3)}</strong><br>
    <span style="font-size:11px;color:#666;">95% CI: ({round(ci_low_rr,3)}, {round(ci_high_rr,3)})</span>
  </div>
  <div style="flex:1; background:#fdf3ee; border-radius:6px; padding:12px 16px; font-size:13px;">
    <strong style="color:#8a4a1a;">Odds Ratio (OR)</strong><br>
    OR = (a × d) ÷ (b × c)<br>
    = ({int(a)} × {int(d)}) ÷ ({int(b)} × {int(c)})<br>
    = {int(a*d)} ÷ {int(b*c)}<br>
    = <strong style="color:#8a4a1a;">{round(or_val,3)}</strong><br>
    <span style="font-size:11px;color:#666;">95% CI: ({round(ci_low_or,3)}, {round(ci_high_or,3)})</span>
  </div>
</div>
"""

    with st.expander(f"🔢 Show me the math — {plabel} & OR"):
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown(f"""
**What does each cell mean?**

| Cell | Meaning |
|------|---------|
| **a** | {row_names[0]} who **have** {col_names[0]} |
| **b** | {row_names[0]} who **do not have** {col_names[0]} |
| **c** | {row_names[1]} who **have** {col_names[0]} |
| **d** | {row_names[1]} who **do not have** {col_names[0]} |

**{plabel} ({pabbr})** compares the {risk_word} in each row: how often does {col_names[0]} occur among {row_names[0]} vs. {row_names[1]}? It uses **row proportions** (a ÷ row total, c ÷ row total).

**Odds Ratio (OR)** compares the *odds* — not the {risk_word} — of {col_names[0]} in each group. Odds = cases ÷ non-cases. The OR uses the **cross-product** (a×d) ÷ (b×c), which is algebraically equivalent to the odds in each row divided.

**When do {pabbr} and OR agree?** When the outcome is rare (<10%), OR ≈ {pabbr}. As the outcome becomes more common, OR diverges further from 1 than {pabbr} does — the OR exaggerates.
        """)

st.title("🧭 Epidemiology Decision Simulator")
col_desc, col_logout = st.columns([6, 1])
with col_desc:
    st.markdown("An interactive epidemiology learning suite — measures of association, advanced epi measures, standardization, hypothesis testing, practice scenarios, and a glossary.")
with col_logout:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Log Out", key="logout_btn"):
        st.session_state["authenticated"] = False
        st.session_state["current_user"] = ""
        st.rerun()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Measures of Association",
    "📐 Advanced Epi Measures",
    "📏 Standardization",
    "🎯 Practice: Measures of Association",
    "🎯 Practice: Advanced Epi Measures",
    "🧪 Hypothesis Testing",
    "📖 Glossary"
])

# ==================================================
# TAB 1: MEASURES OF ASSOCIATION (unchanged logic)
# ==================================================
with tab1:
    PRESETS = {
        "None — I'll enter my own data": None,
        "Cohort: Smoking & Lung Cancer": {
            "design": "Cohort", "outcome_type": "Binary", "exposure_type": "Binary (2 groups)",
            "row_0": "Smoker", "row_1": "Non-smoker", "col_0": "Lung Cancer", "col_1": "No Lung Cancer",
            "cell_0_0": 84, "cell_0_1": 2916, "cell_1_0": 14, "cell_1_1": 2986,
            "description": "**Scenario:** Prospective cohort of 6,000 adults over 10 years. *Doll & Hill (1950).*"
        },
        "Case-Control: H. pylori & Gastric Ulcer": {
            "design": "Case-Control", "outcome_type": "Binary", "exposure_type": "Binary (2 groups)",
            "row_0": "H. pylori positive", "row_1": "H. pylori negative",
            "col_0": "Gastric Ulcer (Case)", "col_1": "No Ulcer (Control)",
            "cell_0_0": 118, "cell_0_1": 62, "cell_1_0": 32, "cell_1_1": 138,
            "description": "**Scenario:** Hospital-based case-control study. *Marshall & Warren (1984).*"
        },
        "Cross-sectional: Obesity & Hypertension": {
            "design": "Cross-sectional", "outcome_type": "Binary", "exposure_type": "Binary (2 groups)",
            "row_0": "Obese (BMI ≥ 30)", "row_1": "Non-obese (BMI < 30)",
            "col_0": "Hypertension", "col_1": "No Hypertension",
            "cell_0_0": 210, "cell_0_1": 290, "cell_1_0": 120, "cell_1_1": 880,
            "description": "**Scenario:** One-time cross-sectional health survey. *NHANES.*"
        },
    }

    if "last_preset" not in st.session_state:
        st.session_state["last_preset"] = None

    col_title, col_reset = st.columns([5, 1])
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_tab1"):
            for k in ["preset_choice","last_preset","design","outcome_type","exposure_type",
                      "row_0","row_1","col_0","col_1","cell_0_0","cell_0_1","cell_1_0","cell_1_1"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

    preset_choice = st.selectbox("Select a scenario:", list(PRESETS.keys()), key="preset_choice")
    preset = PRESETS[preset_choice]

    if preset_choice != st.session_state["last_preset"]:
        st.session_state["last_preset"] = preset_choice
        if preset:
            for key in ["design","outcome_type","exposure_type","row_0","row_1","col_0","col_1",
                        "cell_0_0","cell_0_1","cell_1_0","cell_1_1"]:
                if key in preset: st.session_state[key] = preset[key]
        else:
            defaults = {"row_0":"Group 1","row_1":"Group 2","col_0":"Level 1","col_1":"Level 2",
                       "cell_0_0":0,"cell_0_1":0,"cell_1_0":0,"cell_1_1":0,
                       "design":"Cohort","outcome_type":"Binary","exposure_type":"Binary (2 groups)"}
            for k,v in defaults.items(): st.session_state[k] = v
        st.rerun()

    if preset: st.info(preset["description"])
    st.divider()

    design_options = ["Cohort","Case-Control","Cross-sectional"]
    if "design" not in st.session_state: st.session_state["design"] = "Cohort"
    design = st.selectbox("Select study design:", design_options,
                          index=design_options.index(st.session_state.get("design","Cohort")))

    outcome_options = ["Binary","Categorical (Nominal >2 levels)","Ordinal","Rate (person-time)"]
    if "outcome_type" not in st.session_state: st.session_state["outcome_type"] = "Binary"
    outcome_type = st.selectbox("Select outcome type:", outcome_options,
                                index=outcome_options.index(st.session_state.get("outcome_type","Binary")))

    exposure_options = ["Binary (2 groups)","Categorical (>2 groups)"]
    if "exposure_type" not in st.session_state: st.session_state["exposure_type"] = "Binary (2 groups)"
    exposure_type = st.selectbox("Select exposure type:", exposure_options,
                                 index=exposure_options.index(st.session_state.get("exposure_type","Binary (2 groups)")))

    st.divider()

    if outcome_type in ["Binary","Categorical (Nominal >2 levels)","Ordinal"]:
        num_rows = 2 if exposure_type == "Binary (2 groups)" else st.number_input("Number of Exposure Groups", min_value=2, value=3)
        num_cols = 2 if outcome_type == "Binary" else st.number_input("Number of Outcome Levels", min_value=2, value=3)

        st.subheader("Label Exposure Groups")
        row_names = [st.text_input(f"Exposure Group {i+1}", key=f"row_{i}") for i in range(num_rows)]
        st.subheader("Label Outcome Levels")
        col_names = [st.text_input(f"Outcome Level {j+1}", key=f"col_{j}") for j in range(num_cols)]
        st.subheader("Enter Cell Counts")

        data = []
        for i in range(num_rows):
            st.markdown(f"**{row_names[i]}**")
            row = []
            cols = st.columns(num_cols)
            for j in range(num_cols):
                with cols[j]:
                    row.append(st.number_input(f"{row_names[i]} - {col_names[j]}", min_value=0, key=f"cell_{i}_{j}"))
            data.append(row)

        df = pd.DataFrame(data, columns=col_names, index=row_names)
        df["Row Total"] = df.sum(axis=1)
        total_row = df.sum(); total_row.name = "Column Total"
        df = pd.concat([df, total_row.to_frame().T])
        st.subheader("Contingency Table with Totals")
        st.table(df)
        table_array = df.iloc[:-1, :-1].values

        if st.button("Run Statistical Analysis"):
            if np.sum(table_array) == 0:
                st.warning("⚠️ All cell counts are zero.")
            else:
                row_sums = table_array.sum(axis=1); col_sums = table_array.sum(axis=0)
                if np.any(row_sums == 0) or np.any(col_sums == 0):
                    st.warning("⚠️ One or more rows/columns sum to zero.")
                else:
                    try:
                        chi2, p, dof, expected = chi2_contingency(table_array)
                    except ValueError:
                        st.warning("Chi-square could not be computed.")
                    else:
                        st.subheader("📈 Chi-Square Test of Independence")
                        tail_sel = st.radio("Tail type:", ["Two-tailed (standard for chi-square)","One-tailed (divide p by 2)"], horizontal=True, key="tab1_tail_sel")
                        p_display = p if "Two-tailed" in tail_sel else p / 2
                        tail_note = "" if "Two-tailed" in tail_sel else " (one-tailed)"
                        st.write(f"χ²({dof}) = {round(chi2, 3)}")
                        st.write(f"p-value = {round(p_display,4) if p_display >= 0.0001 else '< 0.0001'}{tail_note}")

                        if p_display < 0.05:
                            st.success(f"Statistically significant association (p = {round(p_display,4)}{tail_note}). We **reject the null hypothesis**.")
                        else:
                            st.warning(f"Insufficient evidence (p = {round(p_display,4)}{tail_note}). We **fail to reject the null hypothesis**.")

                        chi2_explanation_expander(chi2, p_display, dof, table_array, col_names, row_names, tail_note=tail_note)

                        if num_rows == 2 and num_cols == 2:
                            a, b = table_array[0]; c, d = table_array[1]
                            if all(v > 0 for v in [a, b, c, d]):
                                is_cs = (design == "Cross-sectional")
                                plabel = "Prevalence Ratio (PR)" if is_cs else "Risk Ratio (RR)"
                                pabbr = "PR" if is_cs else "RR"
                                rr = (a/(a+b)) / (c/(c+d))
                                se_log_rr = math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+d)))
                                ci_low_rr = math.exp(math.log(rr)-1.96*se_log_rr)
                                ci_high_rr = math.exp(math.log(rr)+1.96*se_log_rr)
                                or_val = (a*d)/(b*c)
                                se_log_or = math.sqrt(1/a+1/b+1/c+1/d)
                                ci_low_or = math.exp(math.log(or_val)-1.96*se_log_or)
                                ci_high_or = math.exp(math.log(or_val)+1.96*se_log_or)

                                st.subheader(plabel)
                                if ci_low_rr <= 1 <= ci_high_rr:
                                    st.warning(f"{pabbr} = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). Not significant.")
                                else:
                                    direction = "higher" if rr > 1 else "lower"
                                    st.success(f"{pabbr} = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). {round(rr,2)}× {direction} risk.")
                                draw_ci(pabbr, rr, ci_low_rr, ci_high_rr)

                                st.subheader("Odds Ratio (OR)")
                                if ci_low_or <= 1 <= ci_high_or:
                                    st.warning(f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). Not significant.")
                                else:
                                    direction = "higher" if or_val > 1 else "lower"
                                    st.success(f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). {round(or_val,2)}× {direction} odds.")
                                draw_ci("OR", or_val, ci_low_or, ci_high_or)
                                rr_or_explanation_expander(a, b, c, d, row_names, col_names,
                                    rr, or_val, ci_low_rr, ci_high_rr, ci_low_or, ci_high_or,
                                    is_cross_sectional=is_cs)

    st.markdown("---")
    st.markdown("Strong epidemiologists think structurally before computing.")

# ==================================================
# TAB 2: ADVANCED EPI MEASURES
# ==================================================
with tab2:
    col_t2, col_r2 = st.columns([5,1])
    with col_r2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_tab2"):
            for k in ["smr_mode","smr_scenario","ar_mode","ar_scenario","nnt_mode","nnt_scenario","hr_mode","hr_scenario"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
    measure = st.selectbox("Select measure to calculate:",
        ["Population Attributable Risk (PAR)","Standardized Mortality Ratio (SMR)",
         "Attributable Risk & AR%","Number Needed to Harm / Treat (NNH/NNT)","Hazard Ratio (HR)"])
    st.divider()

    if measure == "Population Attributable Risk (PAR)":
        st.subheader("Population Attributable Risk (PAR)")
        st.info("PAR estimates the proportion of disease in the **total population** attributable to a specific exposure.")
        data_mode = st.radio("Data entry", ["Use preset scenario","Enter my own data"], horizontal=True)
        if data_mode == "Use preset scenario":
            scenario = st.selectbox("Scenario", ["Smoking & Lung Cancer","Physical Inactivity & T2D","Obesity & CVD"])
            if scenario == "Smoking & Lung Cancer": Pe, RR = 0.14, 15.0
            elif scenario == "Physical Inactivity & T2D": Pe, RR = 0.46, 1.5
            else: Pe, RR = 0.42, 2.0
        else:
            Pe = st.number_input("Exposure prevalence (Pe)", min_value=0.001, max_value=0.999, value=0.30, step=0.01)
            RR = st.number_input("Risk Ratio (RR)", min_value=0.01, value=2.0, step=0.1)
        if st.button("Calculate PAR"):
            PAR_pct = (Pe * (RR - 1)) / (1 + Pe * (RR - 1)) * 100
            col1,col2,col3 = st.columns(3)
            col1.metric("Pe", f"{round(Pe*100,1)}%"); col2.metric("RR", round(RR,2)); col3.metric("PAR%", f"{round(PAR_pct,1)}%")
            st.success(f"{round(PAR_pct,1)}% of all cases in the population are attributable to this exposure.")

    elif measure == "Standardized Mortality Ratio (SMR)":
        st.subheader("Standardized Mortality Ratio (SMR)")
        st.info("SMR = Observed Deaths / Expected Deaths. Compares a study group to a reference population.")
        data_mode = st.radio("Data entry", ["Use preset scenario","Enter my own data"], horizontal=True, key="smr_mode")
        if data_mode == "Use preset scenario":
            scenario = st.selectbox("Scenario", ["Coal Miners & Respiratory Disease","Nuclear Workers & All-Cause Mortality","Firefighters & Cancer"], key="smr_scenario")
            if scenario == "Coal Miners & Respiratory Disease":
                age_groups=["20–34","35–44","45–54","55–64","65–74"]; observed=[2,8,22,41,35]
                ref_rates=[0.0003,0.0010,0.0038,0.0092,0.0198]; pop_sizes=[1200,1800,2100,1600,900]
            elif scenario == "Nuclear Workers & All-Cause Mortality":
                age_groups=["20–34","35–44","45–54","55–64","65–74"]; observed=[3,10,18,29,22]
                ref_rates=[0.0008,0.0018,0.0045,0.0110,0.0240]; pop_sizes=[2000,2500,1800,1200,600]
            else:
                age_groups=["20–34","35–44","45–54","55–64","65–74"]; observed=[1,6,19,38,31]
                ref_rates=[0.0001,0.0006,0.0024,0.0068,0.0160]; pop_sizes=[1500,2000,1900,1400,800]
            smr_df = pd.DataFrame({"Age Group":age_groups,"Pop Size":pop_sizes,"Observed":observed,
                "Ref Rate":ref_rates,"Expected":[round(pop_sizes[i]*ref_rates[i],2) for i in range(5)]})
            st.table(smr_df)
            total_observed = sum(observed); total_expected = sum([pop_sizes[i]*ref_rates[i] for i in range(5)])
        else:
            n_groups = st.number_input("Number of age groups", min_value=1, max_value=10, value=3)
            observed=[]; expected_list=[]
            for i in range(n_groups):
                c1,c2 = st.columns(2)
                with c1: obs = st.number_input(f"Observed {i+1}", min_value=0, key=f"smr_obs_{i}")
                with c2: exp = st.number_input(f"Expected {i+1}", min_value=0.0, step=0.1, key=f"smr_exp_{i}")
                observed.append(obs); expected_list.append(exp)
            total_observed = sum(observed); total_expected = sum(expected_list)
        if st.button("Calculate SMR"):
            if total_expected > 0:
                smr = total_observed / total_expected
                ci_low_s = max(0, smr - 1.96*(smr/math.sqrt(total_observed))) if total_observed > 0 else 0
                ci_high_s = smr + 1.96*(smr/math.sqrt(total_observed)) if total_observed > 0 else 0
                col1,col2,col3 = st.columns(3)
                col1.metric("Observed", int(total_observed)); col2.metric("Expected", round(total_expected,2)); col3.metric("SMR", round(smr,3))
                st.write(f"95% CI: ({round(ci_low_s,3)}, {round(ci_high_s,3)})")
                if ci_low_s <= 1 <= ci_high_s: st.warning("CI includes 1 — not significantly different from reference.")
                elif smr > 1: st.error(f"SMR = {round(smr,2)} — Excess mortality vs. reference population.")
                else: st.success(f"SMR = {round(smr,2)} — Lower mortality. May reflect healthy worker effect.")
                draw_ci("SMR", smr, ci_low_s, ci_high_s)

    elif measure == "Attributable Risk & AR%":
        st.subheader("Attributable Risk (AR) & AR%")
        data_mode = st.radio("Data entry", ["Use preset scenario","Enter my own data"], horizontal=True, key="ar_mode")
        if data_mode == "Use preset scenario":
            scenario = st.selectbox("Scenario", ["Hypertension & CVD","Unvaccinated & Measles","High Sodium & Stroke"], key="ar_scenario")
            if scenario == "Hypertension & CVD": r_exposed, r_unexposed = 0.12, 0.04
            elif scenario == "Unvaccinated & Measles": r_exposed, r_unexposed = 0.90, 0.02
            else: r_exposed, r_unexposed = 0.08, 0.03
        else:
            r_exposed = st.number_input("Risk in exposed", min_value=0.001, max_value=1.0, value=0.12, step=0.01)
            r_unexposed = st.number_input("Risk in unexposed", min_value=0.001, max_value=1.0, value=0.04, step=0.01)
        if st.button("Calculate AR & AR%"):
            ar = r_exposed - r_unexposed; ar_pct = (ar / r_exposed) * 100; rr = r_exposed / r_unexposed
            col1,col2,col3,col4 = st.columns(4)
            col1.metric("Risk (Exposed)", f"{round(r_exposed*100,1)}%"); col2.metric("Risk (Unexposed)", f"{round(r_unexposed*100,1)}%")
            col3.metric("AR", f"{round(ar*100,1)}%"); col4.metric("AR%", f"{round(ar_pct,1)}%")
            st.success(f"AR = {round(ar*100,1)}%: absolute excess risk per 100 exposed people.")
            st.success(f"AR% = {round(ar_pct,1)}%: fraction of disease in the exposed group attributable to the exposure.")

    elif measure == "Number Needed to Harm / Treat (NNH/NNT)":
        st.subheader("NNH / NNT")
        data_mode = st.radio("Data entry", ["Use preset scenario","Enter my own data"], horizontal=True, key="nnt_mode")
        if data_mode == "Use preset scenario":
            scenario = st.selectbox("Scenario", ["Statins & Cardiac Events (NNT)","Aspirin & GI Bleeding (NNH)","Smoking Cessation (NNT)"], key="nnt_scenario")
            if scenario == "Statins & Cardiac Events (NNT)": r_treatment,r_control,label_treatment,label_control = 0.04,0.06,"Statin","Placebo"
            elif scenario == "Aspirin & GI Bleeding (NNH)": r_treatment,r_control,label_treatment,label_control = 0.025,0.010,"Daily aspirin","No aspirin"
            else: r_treatment,r_control,label_treatment,label_control = 0.22,0.08,"Cessation program","No program"
        else:
            label_treatment = st.text_input("Treatment group", "Treatment")
            label_control = st.text_input("Control group", "Control")
            r_treatment = st.number_input(f"Risk ({label_treatment})", min_value=0.001, max_value=1.0, value=0.04, step=0.01)
            r_control = st.number_input(f"Risk ({label_control})", min_value=0.001, max_value=1.0, value=0.06, step=0.01)
        if st.button("Calculate NNT/NNH"):
            risk_diff = abs(r_treatment - r_control)
            if risk_diff > 0:
                nnt = round(1/risk_diff, 1)
                col1,col2,col3 = st.columns(3)
                col1.metric(f"Risk ({label_treatment})", f"{round(r_treatment*100,1)}%")
                col2.metric(f"Risk ({label_control})", f"{round(r_control*100,1)}%")
                col3.metric("Risk Difference", f"{round(risk_diff*100,1)}%")
                if r_treatment < r_control: st.success(f"NNT = {nnt}: treat {nnt} people to prevent one additional outcome.")
                else: st.error(f"NNH = {nnt}: {nnt} people exposed before one additional harm expected.")

    elif measure == "Hazard Ratio (HR)":
        st.subheader("Hazard Ratio (HR)")
        st.info("HR compares the instantaneous event rate over time. Output of Cox proportional hazards regression.")
        data_mode = st.radio("Data entry", ["Use preset scenario","Enter my own data"], horizontal=True, key="hr_mode")
        if data_mode == "Use preset scenario":
            scenario = st.selectbox("Scenario", ["Statins & Time to MI","HIV & Time to AIDS","Physical Activity & Dementia"], key="hr_scenario")
            if scenario == "Statins & Time to MI": hr,ci_low_hr,ci_high_hr,exposed_label,outcome_label = 0.68,0.54,0.85,"Statin therapy","first MI"
            elif scenario == "HIV & Time to AIDS": hr,ci_low_hr,ci_high_hr,exposed_label,outcome_label = 2.31,1.74,3.07,"CD4 < 200","AIDS-defining illness"
            else: hr,ci_low_hr,ci_high_hr,exposed_label,outcome_label = 0.72,0.58,0.89,"High physical activity","dementia"
            col1,col2,col3 = st.columns(3); col1.metric("HR", round(hr,2)); col2.metric("CI Lower", round(ci_low_hr,2)); col3.metric("CI Upper", round(ci_high_hr,2))
            if ci_low_hr <= 1 <= ci_high_hr: st.warning(f"HR = {round(hr,2)} — CI includes 1. Not significant.")
            elif hr < 1: st.success(f"HR = {round(hr,2)}: {exposed_label} had {round((1-hr)*100,1)}% lower hazard of {outcome_label}. Significant.")
            else: st.error(f"HR = {round(hr,2)}: {exposed_label} had {round((hr-1)*100,1)}% higher hazard of {outcome_label}. Significant.")
            draw_ci("HR", hr, ci_low_hr, ci_high_hr)
        else:
            hr = st.number_input("HR", min_value=0.01, value=0.68, step=0.01)
            ci_low_hr = st.number_input("CI Lower", min_value=0.001, value=0.54, step=0.01)
            ci_high_hr = st.number_input("CI Upper", min_value=0.001, value=0.85, step=0.01)
            exposed_label = st.text_input("Exposed group", "Exposed")
            outcome_label = st.text_input("Outcome", "the outcome")
            if st.button("Interpret HR"):
                if ci_low_hr <= 1 <= ci_high_hr: st.warning(f"HR = {round(hr,2)} — CI includes 1. Not significant.")
                elif hr < 1: st.success(f"HR = {round(hr,2)}: {round((1-hr)*100,1)}% lower hazard. Significant.")
                else: st.error(f"HR = {round(hr,2)}: {round((hr-1)*100,1)}% higher hazard. Significant.")
                draw_ci("HR", hr, ci_low_hr, ci_high_hr)

    st.markdown("---")
    st.markdown("Strong epidemiologists think structurally before computing.")

# ==================================================
# TAB 3: STANDARDIZATION
# ==================================================
with tab3:
    col_t3, col_r3 = st.columns([5,1])
    with col_r3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_tab3"):
            for k in ["std_preset_choice"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
    st.markdown("**Standardization** allows fair comparison of rates between populations with different age structures.")
    STD_PRESETS = {
        "None — I'll enter my own data": None,
        "Urban vs. Rural CVD Mortality": {
            "description":"**Scenario:** Compare CVD mortality between an urban (younger) and rural (older) county. Age structure differs substantially — age adjustment reveals the true picture. *Adapted from CDC WONDER.*",
            "age_groups":["0–44","45–54","55–64","65–74","75+"],"std_pop":[150000,40000,35000,25000,15000],
            "pop_a":[80000,15000,12000,8000,4000],"deaths_a":[12,45,120,280,310],
            "pop_b":[30000,18000,22000,20000,14000],"deaths_b":[5,55,145,430,580],
            "label_a":"Urban County","label_b":"Rural County","outcome":"CVD deaths","ref_label":"State population"
        },
        "Coal Miners vs. Office Workers (Lung Disease)": {
            "description":"**Scenario:** Compare lung disease mortality between coal miners and office workers. Miners are older on average — does age account for the difference? *Adapted from NIOSH occupational health data.*",
            "age_groups":["20–34","35–44","45–54","55–64","65–74"],"std_pop":[5000,6000,5500,4000,2000],
            "pop_a":[800,1800,2100,1600,900],"deaths_a":[1,6,18,38,32],
            "pop_b":[2000,2200,1800,1200,600],"deaths_b":[0,2,5,10,8],
            "label_a":"Coal Miners","label_b":"Office Workers","outcome":"lung disease deaths","ref_label":"Workforce population"
        },
        "State A vs. State B: Diabetes Mortality": {
            "description":"**Scenario:** Two states with different age distributions are being compared on diabetes mortality rates. A policy team wants to know whether the difference reflects true disease burden or age structure. *Hypothetical scenario based on CDC patterns.*",
            "age_groups":["0–44","45–54","55–64","65–74","75+"],"std_pop":[200000,55000,48000,35000,22000],
            "pop_a":[420000,80000,65000,42000,18000],"deaths_a":[8,62,180,390,420],
            "pop_b":[180000,55000,62000,58000,45000],"deaths_b":[4,48,198,620,890],
            "label_a":"State A","label_b":"State B","outcome":"diabetes deaths","ref_label":"National population"
        },
    }
    std_preset_choice = st.selectbox("Load a preset:", list(STD_PRESETS.keys()), key="std_preset_choice")
    std_preset = STD_PRESETS[std_preset_choice]
    if std_preset: st.info(std_preset["description"])
    st.divider()

    if std_preset:
        age_groups=std_preset["age_groups"]; std_pop=std_preset["std_pop"]
        pop_a=std_preset["pop_a"]; deaths_a=std_preset["deaths_a"]
        pop_b=std_preset["pop_b"]; deaths_b=std_preset["deaths_b"]
        label_a=std_preset["label_a"]; label_b=std_preset["label_b"]
        outcome_lbl=std_preset["outcome"]; ref_label=std_preset["ref_label"]; n_groups=len(age_groups)
    else:
        col1,col2 = st.columns(2)
        with col1: label_a=st.text_input("Population A","Population A"); label_b=st.text_input("Population B","Population B")
        with col2: ref_label=st.text_input("Reference population","Standard Population"); outcome_lbl=st.text_input("Outcome","deaths")
        n_groups = st.number_input("Number of age groups", min_value=2, max_value=10, value=5)
        age_groups,std_pop,pop_a,deaths_a,pop_b,deaths_b = [],[],[],[],[],[]
        for i in range(int(n_groups)):
            cols = st.columns([2,2,2,2,2,2])
            age_groups.append(cols[0].text_input("",f"Group {i+1}",key=f"ag_{i}",label_visibility="collapsed"))
            std_pop.append(cols[1].number_input("",min_value=1,value=10000,key=f"sp_{i}",label_visibility="collapsed"))
            pop_a.append(cols[2].number_input("",min_value=1,value=1000,key=f"pa_{i}",label_visibility="collapsed"))
            deaths_a.append(cols[3].number_input("",min_value=0,value=0,key=f"da_{i}",label_visibility="collapsed"))
            pop_b.append(cols[4].number_input("",min_value=1,value=1000,key=f"pb_{i}",label_visibility="collapsed"))
            deaths_b.append(cols[5].number_input("",min_value=0,value=0,key=f"db_{i}",label_visibility="collapsed"))

    if st.button("Run Standardization Analysis"):
        if sum(pop_a) == 0 or sum(pop_b) == 0:
            st.warning("Population sizes cannot be zero.")
        else:
            rate_a = [deaths_a[i]/max(pop_a[i],1)*100000 for i in range(n_groups)]
            rate_b = [deaths_b[i]/max(pop_b[i],1)*100000 for i in range(n_groups)]
            ref_rate = [(deaths_a[i]+deaths_b[i])/max(pop_a[i]+pop_b[i],1)*100000 for i in range(n_groups)]
            expected_a_direct = [rate_a[i]/100000*std_pop[i] for i in range(n_groups)]
            expected_b_direct = [rate_b[i]/100000*std_pop[i] for i in range(n_groups)]
            age_adj_rate_a = sum(expected_a_direct)/sum(std_pop)*100000
            age_adj_rate_b = sum(expected_b_direct)/sum(std_pop)*100000
            crude_rate_a = sum(deaths_a)/sum(pop_a)*100000
            crude_rate_b = sum(deaths_b)/sum(pop_b)*100000
            crude_higher = label_a if crude_rate_a > crude_rate_b else label_b
            adj_higher = label_a if age_adj_rate_a > age_adj_rate_b else label_b

            col1,col2 = st.columns(2)
            with col1:
                st.markdown(f"### {label_a}")
                st.metric("Crude Rate (per 100,000)", round(crude_rate_a,1))
                st.metric("Age-Adjusted Rate", round(age_adj_rate_a,1))
            with col2:
                st.markdown(f"### {label_b}")
                st.metric("Crude Rate (per 100,000)", round(crude_rate_b,1))
                st.metric("Age-Adjusted Rate", round(age_adj_rate_b,1))

            if crude_higher != adj_higher:
                st.error("⚠️ **Confounding by age detected!** Crude and age-adjusted rates point in opposite directions.")
            else:
                st.success("✅ Age structure had minimal impact. Crude and adjusted rates tell a similar story.")

    st.markdown("---")
    st.markdown("Strong epidemiologists think structurally before computing.")

# ==================================================
# TAB 4: PRACTICE — MEASURES OF ASSOCIATION
# QUICK WIN 1: Submit button, locked answers, "what you missed"
# ==================================================
with tab4:
    st.markdown("""
    Read each scenario, make **all three decisions**, then click **Submit My Answers**.
    Feedback is hidden until you commit — work through your reasoning first.
    """)

    PRACTICE_SCENARIOS = [
        {
            "id": "s1", "title": "Scenario 1: Lead Exposure & Cognitive Development",
            "description": "400 children near a lead smelting plant and 400 from unexposed neighborhoods are followed for 3 years. New learning disability diagnoses are recorded.",
            "correct_design": "Cohort", "correct_outcome": "Binary", "correct_exposure": "Binary (2 groups)",
            "design_hint": "Researchers classified children by **exposure status**, then tracked who developed a new diagnosis. Grouping by exposure and following to outcome — regardless of whether data were collected prospectively or retrospectively — is the hallmark of a cohort study.",
            "outcome_hint": "Learning disability: present or absent — **two categories = binary**.",
            "exposure_hint": "Exposed vs. unexposed neighborhoods — **two groups = binary**.",
            "design_wrong": {
                "Case-Control": "❌ A case-control study starts with people who **already have** the disease, then looks backward at past exposure. Here researchers started with exposure status (lead vs. no lead) and tracked who developed a diagnosis — that's cohort.",
                "Cross-sectional": "❌ A cross-sectional study measures exposure and outcome **at the same point in time**. Here children were followed over 3 years to capture new diagnoses — the separation between exposure classification and outcome measurement is what makes this a cohort."
            },
            "outcome_wrong": {
                "Categorical (Nominal >2 levels)": "❌ Categorical requires **3+ unordered categories**. Diagnosis is present or absent — two categories — binary.",
                "Ordinal": "❌ Ordinal requires **ordered categories**. A diagnosis is yes or no — binary.",
                "Rate (person-time)": "❌ Rate outcomes are used when follow-up **varies across participants**. All children were followed for the same 3-year period — binary outcome works."
            },
            "exposure_wrong": {"Categorical (>2 groups)": "❌ Categorical requires **3+ groups**. There are only two groups here — binary."},
            "data": {"type": "contingency", "context": "3-year follow-up data. Calculate RR, OR, and p-value.",
                     "row_names": ["Lead-exposed","Unexposed"], "col_names": ["Learning Disability","No Learning Disability"],
                     "cells": [[52,348],[21,379]]}
        },
        {
            "id": "s2", "title": "Scenario 2: Fast Food & Obesity",
            "description": "One-time survey of 2,500 adults. Participants report weekly fast food frequency (never/1–2x/3–4x/5+x) and BMI is measured to classify obesity (BMI ≥ 30 vs. < 30).",
            "correct_design": "Cross-sectional", "correct_outcome": "Binary", "correct_exposure": "Categorical (>2 groups)",
            "design_hint": "**One-time survey** — both exposure and outcome measured simultaneously. No follow-up = cross-sectional.",
            "outcome_hint": "Obesity: BMI ≥ 30 vs. < 30 — **two categories = binary**.",
            "exposure_hint": "Four frequency categories — **more than 2 groups = categorical**.",
            "design_wrong": {
                "Cohort": "❌ Cohort follows people **over time**. This is a one-time survey — no follow-up period.",
                "Case-Control": "❌ Case-control recruits by disease status and looks back. This survey recruited everyone at once and measured everything simultaneously."
            },
            "outcome_wrong": {
                "Categorical (Nominal >2 levels)": "❌ Obesity is simply present or absent — two categories. Categorical requires 3+.",
                "Ordinal": "❌ Obesity (yes/no) is two categories — binary.",
                "Rate (person-time)": "❌ No follow-up time variation — this is a one-time survey."
            },
            "exposure_wrong": {"Binary (2 groups)": "❌ Four frequency categories (never/1–2x/3–4x/5+x) — more than 2 groups = categorical."},
            "data": {"type": "contingency_wide", "context": "Survey data by fast food frequency and obesity.",
                     "row_names": ["Never","1–2x/week","3–4x/week","5+x/week"], "col_names": ["Obese","Not Obese"],
                     "cells": [[62,538],[118,682],[189,561],[141,209]]}
        },
        {
            "id": "s3", "title": "Scenario 3: HPV Vaccine & Cervical Cancer",
            "description": "250 women with cervical cancer and 500 without are recruited. Vaccination history is assessed from medical records.",
            "correct_design": "Case-Control", "correct_outcome": "Binary", "correct_exposure": "Binary (2 groups)",
            "design_hint": "Started with **disease status** (cases vs. controls), then looked **backward** at vaccination. Starting with outcome and looking back = case-control.",
            "outcome_hint": "Cervical cancer: present or absent — **binary**.",
            "exposure_hint": "Vaccinated vs. unvaccinated — **two groups = binary**.",
            "design_wrong": {
                "Cohort": "❌ A cohort study would classify women by **vaccination status** and then track who developed cervical cancer. Here researchers started with cancer status (cases vs. controls) and looked back at vaccination history — that's case-control.",
                "Cross-sectional": "❌ Cross-sectional measures simultaneously. Here researchers specifically recruited cases and controls by disease status and looked back — case-control."
            },
            "outcome_wrong": {
                "Categorical (Nominal >2 levels)": "❌ Cervical cancer is present or absent — binary.",
                "Ordinal": "❌ A diagnosis is yes or no — binary.",
                "Rate (person-time)": "❌ In case-control, the outcome is already determined before the study begins."
            },
            "exposure_wrong": {"Categorical (>2 groups)": "❌ Vaccinated vs. unvaccinated — two groups only = binary."},
            "data": {"type": "contingency", "context": "Case-control data. Odds Ratio is the appropriate measure.",
                     "row_names": ["Unvaccinated","Vaccinated"], "col_names": ["Cervical Cancer (Case)","No Cancer (Control)"],
                     "cells": [[178,182],[72,318]]}
        },
        {
            "id": "s4", "title": "Scenario 4: Shift Work & Metabolic Syndrome",
            "description": "1,200 hospital employees classified by shift: day only, rotating, or night. Followed 5 years. Metabolic syndrome (yes/no) assessed at end.",
            "correct_design": "Cohort", "correct_outcome": "Binary", "correct_exposure": "Categorical (>2 groups)",
            "design_hint": "Employees were classified by **exposure (shift type)** and followed to see who developed metabolic syndrome — the defining logic of a cohort study.",
            "outcome_hint": "Metabolic syndrome: present or absent — **binary**.",
            "exposure_hint": "Three shift types — **more than 2 groups = categorical**.",
            "design_wrong": {
                "Case-Control": "❌ A case-control study would start by recruiting people **with metabolic syndrome** (cases) and those without (controls), then look back at shift history. Here employees were classified by shift type first and the outcome was measured afterward — that's cohort.",
                "Cross-sectional": "❌ Employees were followed for **5 years** — that follow-up period makes this cohort, not cross-sectional."
            },
            "outcome_wrong": {
                "Categorical (Nominal >2 levels)": "❌ Metabolic syndrome is present or absent — binary.",
                "Ordinal": "❌ A yes/no diagnosis is binary.",
                "Rate (person-time)": "❌ All employees followed the same 5-year period — no need for person-time."
            },
            "exposure_wrong": {"Binary (2 groups)": "❌ Three categories (day/rotating/night) — more than 2 = categorical."},
            "data": {"type": "contingency_wide", "context": "5-year follow-up data by shift type.",
                     "row_names": ["Day shift","Rotating shift","Night shift"], "col_names": ["Metabolic Syndrome","No Metabolic Syndrome"],
                     "cells": [[62,338],[98,302],[121,279]]}
        },
        {
            "id": "s5", "title": "Scenario 5: Air Pollution & Emergency Department Visits",
            "description": "3,000 adults monitored for PM2.5 over 2 years. Participants move and vary in outdoor time — each contributes different observation time. Outcome: new ED visits for respiratory illness.",
            "correct_design": "Cohort", "correct_outcome": "Rate (person-time)", "correct_exposure": "Binary (2 groups)",
            "design_hint": "Participants were classified by **PM2.5 exposure level** and tracked to see who developed ED visits — exposure grouping → outcome measurement is the cohort logic.",
            "outcome_hint": "Each person contributes **different follow-up time** — must account for this using person-time. Rate outcome.",
            "exposure_hint": "High vs. low PM2.5 — **two groups = binary**.",
            "design_wrong": {
                "Case-Control": "❌ A case-control study would start by recruiting people **who already had ED visits** (cases) and look back at past pollution exposure. Here participants were classified by exposure level first and then followed for new events — that's cohort.",
                "Cross-sectional": "❌ Followed over 2 years — not a one-time snapshot."
            },
            "outcome_wrong": {
                "Binary": "❌ Follow-up time **varies** across participants. Simply counting who got sick ignores this variation — you need person-time to create a fair rate.",
                "Categorical (Nominal >2 levels)": "❌ Event counts per varying follow-up time is a rate, not unordered categories.",
                "Ordinal": "❌ Not ordered categories — a rate per person-time."
            },
            "exposure_wrong": {"Categorical (>2 groups)": "❌ High vs. low — two groups = binary."},
            "data": {"type": "rate", "context": "Person-time data. Calculate IRR.",
                     "row_names": ["High PM2.5","Low PM2.5"], "cases": [187,64], "person_time": [4200,5100]}
        },
        {
            "id": "s6", "title": "Scenario 6: Food Insecurity & Depression",
            "description": "One-time survey of 5,000 households. Food insecurity (secure vs. insecure) and PHQ-9 depression scores (no/mild/moderate/severe depression) are measured simultaneously.",
            "correct_design": "Cross-sectional", "correct_outcome": "Categorical (Nominal >2 levels)", "correct_exposure": "Binary (2 groups)",
            "design_hint": "Both measured **simultaneously in a single survey** — cross-sectional.",
            "outcome_hint": "Four depression categories — **more than 2 = categorical**.",
            "exposure_hint": "Food secure vs. insecure — **two groups = binary**.",
            "design_wrong": {
                "Cohort": "❌ Single survey — no follow-up period. Cross-sectional.",
                "Case-Control": "❌ Did not recruit by disease status — surveyed a random sample simultaneously."
            },
            "outcome_wrong": {
                "Binary": "❌ Four categories (none/mild/moderate/severe) — not binary. Binary would require collapsing to just depressed vs. not depressed.",
                "Ordinal": "❌ These categories are ordered, which makes them technically ordinal. However, the app treats ordinal as categorical for chi-square. **Either answer is defensible** — the key insight is it's not binary.",
                "Rate (person-time)": "❌ One-time survey — no follow-up time."
            },
            "exposure_wrong": {"Categorical (>2 groups)": "❌ Food secure vs. insecure — two groups = binary."},
            "data": {"type": "contingency_wide", "context": "Survey data: depression severity by food insecurity.",
                     "row_names": ["Food Insecure","Food Secure"], "col_names": ["No Depression","Mild","Moderate","Severe"],
                     "cells": [[312,284,198,106],[2180,980,412,28]]}
        },
        {
            "id": "s7", "title": "Scenario 7: Air Pollution Spikes & Myocardial Infarction",
            "description": "Researchers identify 2,100 patients who were admitted to hospital with a confirmed myocardial infarction. For each patient, they compare PM2.5 air pollution levels in the hour before symptom onset (the hazard period) to PM2.5 levels at the same time of day one week earlier for the same patient (the control period). No separate control group is recruited.",
            "correct_design": "Case-Crossover",
            "correct_outcome": "Binary",
            "correct_exposure": "Binary (2 groups)",
            "design_hint": "Each MI patient is compared to **themselves** at a different time — no separate control group. The hazard period (just before the event) is contrasted with a control period (same person, no event). This self-matched structure is the defining feature of a **case-crossover** design.",
            "outcome_hint": "MI occurred or did not occur during the hazard period — **two categories = binary**.",
            "exposure_hint": "High vs. low PM2.5 exposure — **two groups = binary**.",
            "design_wrong": {
                "Cohort": "❌ A cohort study would group people by PM2.5 exposure level and follow them forward to see who had an MI. Here everyone already had an MI — there is no unexposed comparison group followed over time.",
                "Case-Control": "❌ A standard case-control recruits a **separate control group** of people without the disease. Here there are no external controls — each case is compared to themselves at a different time. That self-matching is case-crossover.",
                "Cross-sectional": "❌ Cross-sectional measures exposure and outcome simultaneously at one time point. Here researchers are comparing exposure across two time periods for the same person — that's case-crossover."
            },
            "outcome_wrong": {
                "Categorical (Nominal >2 levels)": "❌ The outcome is MI: occurred or did not occur — two categories = binary.",
                "Ordinal": "❌ MI is yes or no — binary.",
                "Rate (person-time)": "❌ The comparison here is between two time windows for the same person, not varying follow-up time across participants."
            },
            "exposure_wrong": {"Categorical (>2 groups)": "❌ PM2.5 is classified as high vs. low — two groups = binary."},
            "data": {"type": "contingency", "context": "Matched exposure data. OR is the appropriate measure — each person is their own control.",
                     "row_names": ["High PM2.5 (hazard period)","Low PM2.5 (hazard period)"],
                     "col_names": ["High PM2.5 (control period)","Low PM2.5 (control period)"],
                     "cells": [[210, 480],[95, 1315]]}
        },
        {
            "id": "s8", "title": "Scenario 8: Alcohol Consumption & Occupational Injury",
            "description": "An occupational health team recruits 850 workers who sustained a workplace injury requiring medical attention. Each worker is asked about alcohol consumption in the 6 hours before the injury (the hazard period) and alcohol consumption during the same 6-hour window on a workday one week prior (the control period) — for the same worker. No non-injured workers are recruited.",
            "correct_design": "Case-Crossover",
            "correct_outcome": "Binary",
            "correct_exposure": "Binary (2 groups)",
            "design_hint": "Injured workers are compared to **themselves** at a prior time — same person, different exposure window. No separate control group of uninjured workers. Self-matched comparison across time periods = **case-crossover**.",
            "outcome_hint": "Injury: occurred or did not occur — **binary**.",
            "exposure_hint": "Alcohol consumed vs. not consumed in the 6-hour window — **two groups = binary**.",
            "design_wrong": {
                "Cohort": "❌ A cohort would follow workers from alcohol exposure forward to see who got injured. Here everyone is already injured — the comparison is between time periods for the same person.",
                "Case-Control": "❌ Standard case-control would recruit uninjured workers as a separate control group. Here there are no external controls — each injured worker is their own control across two time windows. That's case-crossover.",
                "Cross-sectional": "❌ Cross-sectional captures a single moment. Here each worker contributes two time windows — a hazard period and a control period — and exposure is compared across them."
            },
            "outcome_wrong": {
                "Categorical (Nominal >2 levels)": "❌ Injury occurred or did not — two categories = binary.",
                "Ordinal": "❌ Injury is yes or no — binary.",
                "Rate (person-time)": "❌ The comparison is between two defined time windows per person, not varying follow-up time."
            },
            "exposure_wrong": {"Categorical (>2 groups)": "❌ Alcohol consumed vs. not consumed — two groups = binary."},
            "data": {"type": "contingency", "context": "Matched exposure data. OR is appropriate — each worker is their own control.",
                     "row_names": ["Alcohol: Yes (hazard period)","Alcohol: No (hazard period)"],
                     "col_names": ["Alcohol: Yes (control period)","Alcohol: No (control period)"],
                     "cells": [[38, 156],[47, 609]]}
        },
    ]

    design_options   = ["— Select —","Cohort","Case-Control","Cross-sectional","Case-Crossover"]
    outcome_options  = ["— Select —","Binary","Categorical (Nominal >2 levels)","Ordinal","Rate (person-time)"]
    exposure_options = ["— Select —","Binary (2 groups)","Categorical (>2 groups)"]

    if "prac_scenario_order" not in st.session_state:
        order = list(range(len(PRACTICE_SCENARIOS))); random.shuffle(order)
        st.session_state["prac_scenario_order"] = order
    SHUFFLED_PRACTICE = [PRACTICE_SCENARIOS[i] for i in st.session_state["prac_scenario_order"]]

    col_hdr, col_rst = st.columns([5,1])
    with col_hdr: st.caption(f"**{len(PRACTICE_SCENARIOS)} scenarios** — randomized order. Reset to shuffle.")
    with col_rst:
        if st.button("🔄 Reset", key="reset_tab4"):
            for sc in PRACTICE_SCENARIOS:
                for f in ["design","outcome","exposure","submitted"]:
                    k = f"prac_{sc['id']}_{f}"
                    if k in st.session_state: del st.session_state[k]
            if "prac_scenario_order" in st.session_state: del st.session_state["prac_scenario_order"]
            st.rerun()

    for sc in SHUFFLED_PRACTICE:
        st.divider()
        st.subheader(sc["title"])
        st.markdown(sc["description"])
        sid = sc["id"]
        submitted_key = f"prac_{sid}_submitted"
        already_submitted = st.session_state.get(submitted_key, False)

        design_choice = st.selectbox("What is the study design?", design_options, key=f"prac_{sid}_design", disabled=already_submitted)
        outcome_choice = st.selectbox("What is the outcome variable type?", outcome_options, key=f"prac_{sid}_outcome", disabled=already_submitted)
        exposure_choice = st.selectbox("What is the exposure variable type?", exposure_options, key=f"prac_{sid}_exposure", disabled=already_submitted)

        all_selected = all(st.session_state.get(f"prac_{sid}_{f}") not in [None,"— Select —"] for f in ["design","outcome","exposure"])

        if not already_submitted:
            if all_selected:
                if st.button("Submit My Answers", key=f"submit_{sid}", type="primary"):
                    st.session_state[submitted_key] = True; st.rerun()
            else:
                st.caption("⬆️ Make all three selections before submitting.")

        if already_submitted:
            dv = st.session_state.get(f"prac_{sid}_design")
            ov = st.session_state.get(f"prac_{sid}_outcome")
            ev = st.session_state.get(f"prac_{sid}_exposure")
            dc = dv == sc["correct_design"]; oc = ov == sc["correct_outcome"]; ec = ev == sc["correct_exposure"]
            all_correct = dc and oc and ec
            correct_count = sum([dc, oc, ec])

            if not all_correct:
                st.error(f"📋 **{correct_count}/3 correct — here's what you missed:**")
                if not dc:
                    wrong_hint = sc.get("design_wrong",{}).get(dv,"")
                    if wrong_hint: st.markdown(f"**Study Design** — You selected: *{dv}*\n\n{wrong_hint}")
                    st.markdown(f"✅ **Correct:** {sc['correct_design']} — {sc['design_hint']}")
                if not oc:
                    wrong_hint = sc.get("outcome_wrong",{}).get(ov,"")
                    if wrong_hint: st.markdown(f"**Outcome Type** — You selected: *{ov}*\n\n{wrong_hint}")
                    st.markdown(f"✅ **Correct:** {sc['correct_outcome']} — {sc['outcome_hint']}")
                if not ec:
                    wrong_hint = sc.get("exposure_wrong",{}).get(ev,"")
                    if wrong_hint: st.markdown(f"**Exposure Type** — You selected: *{ev}*\n\n{wrong_hint}")
                    st.markdown(f"✅ **Correct:** {sc['correct_exposure']} — {sc['exposure_hint']}")
                if st.button("🔄 Try Again", key=f"retry_{sid}"):
                    for f in ["design","outcome","exposure","submitted"]:
                        k = f"prac_{sid}_{f}"
                        if k in st.session_state: del st.session_state[k]
                    st.rerun()
            else:
                st.success("🎯 Perfect — all three correct!")
                st.markdown(f"**Design:** {sc['correct_design']} — {sc['design_hint']}")
                st.markdown(f"**Outcome:** {sc['correct_outcome']} — {sc['outcome_hint']}")
                st.markdown(f"**Exposure:** {sc['correct_exposure']} — {sc['exposure_hint']}")

            if all_correct and "data" in sc:
                st.markdown("---")
                st.markdown("### 📋 Now run the analysis")
                st.markdown(sc["data"]["context"])
                d = sc["data"]

                if d["type"] in ["contingency","contingency_wide"]:
                    df_d = pd.DataFrame(d["cells"], columns=d["col_names"], index=d["row_names"])
                    df_d["Row Total"] = df_d.sum(axis=1)
                    tr = df_d.sum(); tr.name = "Column Total"
                    df_d = pd.concat([df_d, tr.to_frame().T]); st.table(df_d)
                    if st.button("Run Statistical Analysis", key=f"run_{sid}"):
                        table = np.array(d["cells"]); chi2_val, p_val, dof, _ = chi2_contingency(table)
                        st.write(f"χ²({dof}) = {round(chi2_val,3)}, p = {round(p_val,4) if p_val >= 0.0001 else '< 0.0001'}")
                        if p_val < 0.05: st.success("Statistically significant. Reject H₀.")
                        else: st.warning("Insufficient evidence. Fail to reject H₀.")
                        chi2_explanation_expander(chi2_val, p_val, dof, table, d["col_names"], d["row_names"])
                        if d["type"] == "contingency":
                            a,b = table[0]; c,dd = table[1]
                            if all(v > 0 for v in [a,b,c,dd]):
                                rr=(a/(a+b))/(c/(c+dd)); se_log_rr=math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+dd)))
                                ci_low_rr=math.exp(math.log(rr)-1.96*se_log_rr); ci_high_rr=math.exp(math.log(rr)+1.96*se_log_rr)
                                or_val=(a*dd)/(b*c); se_log_or=math.sqrt(1/a+1/b+1/c+1/dd)
                                ci_low_or=math.exp(math.log(or_val)-1.96*se_log_or); ci_high_or=math.exp(math.log(or_val)+1.96*se_log_or)
                                st.subheader("Risk Ratio (RR)")
                                if ci_low_rr <= 1 <= ci_high_rr: st.warning(f"RR = {round(rr,2)} — Not significant.")
                                else:
                                    direction = "higher" if rr > 1 else "lower"
                                    st.success(f"RR = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). {round(rr,2)}× {direction}.")
                                draw_ci("RR", rr, ci_low_rr, ci_high_rr)
                                st.subheader("Odds Ratio (OR)")
                                if ci_low_or <= 1 <= ci_high_or: st.warning(f"OR = {round(or_val,2)} — Not significant.")
                                else:
                                    direction = "higher" if or_val > 1 else "lower"
                                    st.success(f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). {round(or_val,2)}× {direction}.")
                                draw_ci("OR", or_val, ci_low_or, ci_high_or)
                                rr_or_explanation_expander(a, b, c, dd, d["row_names"], d["col_names"],
                                    rr, or_val, ci_low_rr, ci_high_rr, ci_low_or, ci_high_or)
                        else:
                            st.info("With 3+ categories, chi-square is the appropriate test. RR/OR require a 2×2 table.")

                elif d["type"] == "rate":
                    df_d = pd.DataFrame({"Group":d["row_names"],"Cases":d["cases"],"Person-Time":d["person_time"],
                        "Rate per 100k":[round(d["cases"][i]/d["person_time"][i]*100000,1) for i in range(len(d["cases"]))]})
                    st.table(df_d)
                    if st.button("Run Statistical Analysis", key=f"run_{sid}"):
                        c1,c2=d["cases"]; pt1,pt2=d["person_time"]
                        irr=(c1/pt1)/(c2/pt2); se_log_irr=math.sqrt((1/c1)+(1/c2))
                        ci_low_irr=math.exp(math.log(irr)-1.96*se_log_irr); ci_high_irr=math.exp(math.log(irr)+1.96*se_log_irr)
                        st.write(f"IRR = {round(irr,3)}, 95% CI: ({round(ci_low_irr,3)}, {round(ci_high_irr,3)})")
                        if ci_low_irr <= 1 <= ci_high_irr: st.warning("CI includes 1. Not significant.")
                        else:
                            direction = "higher" if irr > 1 else "lower"
                            st.success(f"IRR = {round(irr,2)} — Rate in {d['row_names'][0]} is {round(irr,2)}× {direction}.")
                        draw_ci("IRR", irr, ci_low_irr, ci_high_irr)

    st.divider()
    total_correct=0; answered=0
    for sc in PRACTICE_SCENARIOS:
        sid=sc["id"]
        if st.session_state.get(f"prac_{sid}_submitted"):
            answered+=3
            total_correct+=sum([st.session_state.get(f"prac_{sid}_design")==sc["correct_design"],
                                 st.session_state.get(f"prac_{sid}_outcome")==sc["correct_outcome"],
                                 st.session_state.get(f"prac_{sid}_exposure")==sc["correct_exposure"]])
    if answered > 0:
        pct = round(total_correct/answered*100)
        st.subheader(f"📊 Score: {total_correct} / {answered}")
        st.progress(pct/100)
        if pct==100: st.success("🏆 Perfect on all submitted scenarios!")
        elif pct>=75: st.info("Good work — review any missed scenarios.")
        else: st.warning("Review the feedback above and try again.")

    st.markdown("---")
    st.markdown("Strong epidemiologists think structurally before computing.")

# ==================================================
# TAB 5: PRACTICE — ADVANCED EPI MEASURES
# QUICK WIN 1 (continued): Submit button, locked, "what you missed"
# ==================================================
with tab5:
    st.markdown("""
    Select the most appropriate advanced measure for each scenario, then click **Submit My Answer**.
    Feedback is hidden until you commit.
    """)

    ADV_SCENARIOS = [
        {
            "id": "adv_1", "title": "Scenario 1: Obesity & Coronary Heart Disease",
            "description": "A public health analyst wants to estimate how much CHD burden could be eliminated if obesity were eradicated. 42% of US adults have obesity; they have 1.8× the risk of CHD.",
            "correct_measure": "Population Attributable Risk (PAR)",
            "measure_hint": "The question is about **population-level preventable burden** — what fraction of all CHD in the US could be prevented. PAR combines exposure prevalence (Pe) with RR to answer this.",
            "measure_wrong": {
                "Standardized Mortality Ratio (SMR)": "❌ SMR compares observed to expected deaths vs. a reference population. This question asks about population-level preventable fraction — that's PAR.",
                "Attributable Risk & AR%": "❌ AR% estimates the fraction within the **exposed group** only. PAR goes further — it estimates fraction across the **total population**, accounting for how common obesity is.",
                "Number Needed to Harm / Treat (NNH/NNT)": "❌ NNT/NNH express per-person benefit or harm. This question asks about a population-level fraction — PAR.",
                "Hazard Ratio (HR)": "❌ HR compares event rates over time. This question asks what fraction of all CHD is attributable to obesity — PAR."
            },
            "data": {"type": "par", "context": "Calculate PAR% — fraction of CHD attributable to obesity.", "Pe": 0.42, "RR": 1.8}
        },
        {
            "id": "adv_2", "title": "Scenario 2: Rubber Workers & Bladder Cancer",
            "description": "4,200 rubber workers followed 15 years. 38 developed bladder cancer. Applying general population rates to this cohort's age structure predicts only 18.4 expected cases.",
            "correct_measure": "Standardized Mortality Ratio (SMR)",
            "measure_hint": "You have **observed cases (38) and expected cases (18.4)** derived from applying reference population rates. Observed ÷ Expected = SMR. Comparing one group to expected counts from a reference = SMR.",
            "measure_wrong": {
                "Population Attributable Risk (PAR)": "❌ PAR requires knowing exposure prevalence in the general population and an RR. Here you have observed vs. expected counts — SMR.",
                "Attributable Risk & AR%": "❌ AR% compares two groups within the same study. Here you're comparing one occupational group to a reference population — SMR.",
                "Number Needed to Harm / Treat (NNH/NNT)": "❌ NNH requires a risk difference between two groups. Here you have one group's observed vs. expected counts — SMR.",
                "Hazard Ratio (HR)": "❌ HR requires Cox regression with time-to-event data. Here you simply have total observed vs. expected counts — SMR."
            },
            "data": {"type": "smr", "context": "Calculate SMR.", "observed": 38, "expected": 18.4, "group_label": "Rubber workers", "outcome_label": "bladder cancer"}
        },
        {
            "id": "adv_3", "title": "Scenario 3: Hypertension & Stroke",
            "description": "14% of uncontrolled hypertension patients had a stroke over 10 years vs. 4% of controlled patients. Of all strokes in the uncontrolled group, what fraction is due to uncontrolled BP?",
            "correct_measure": "Attributable Risk & AR%",
            "measure_hint": "The question asks about the **fraction of disease within the exposed group** (uncontrolled hypertension) attributable to the exposure. That's exactly AR%.",
            "measure_wrong": {
                "Population Attributable Risk (PAR)": "❌ PAR estimates the fraction across the **entire population**. The question asks specifically about the fraction within the **exposed group** — AR%.",
                "Standardized Mortality Ratio (SMR)": "❌ SMR compares to a reference population. This compares two groups within the same study — AR%.",
                "Number Needed to Harm / Treat (NNH/NNT)": "❌ NNT tells how many need treatment to prevent one event. The question asks what fraction of strokes is attributable to exposure — AR%.",
                "Hazard Ratio (HR)": "❌ HR uses Cox regression. This uses 10-year cumulative risks to assess attributable fraction — AR%."
            },
            "data": {"type": "ar", "context": "Calculate AR and AR%.", "r_exposed": 0.14, "r_unexposed": 0.04,
                     "exposed_label": "Uncontrolled hypertension", "unexposed_label": "Controlled hypertension"}
        },
        {
            "id": "adv_4", "title": "Scenario 4: Naloxone Programs & Overdose Deaths",
            "description": "3% of communities with naloxone programs had overdose deaths vs. 7% without. How many communities need the program to prevent one additional overdose death?",
            "correct_measure": "Number Needed to Harm / Treat (NNH/NNT)",
            "measure_hint": "**'How many need the intervention to prevent one event'** is the definition of NNT. It's the most intuitive way to communicate benefit to policymakers. NNT = 1 / Risk Difference.",
            "measure_wrong": {
                "Population Attributable Risk (PAR)": "❌ PAR estimates population-level preventable fraction. The question asks how many need treatment to prevent one event — NNT.",
                "Standardized Mortality Ratio (SMR)": "❌ SMR compares to a reference population. The question asks for a per-community treatment benefit — NNT.",
                "Attributable Risk & AR%": "❌ AR% estimates the fraction attributable in the exposed group. The question asks for an intuitive per-community figure — NNT.",
                "Hazard Ratio (HR)": "❌ HR uses time-to-event Cox regression. Simple cumulative proportions + per-treatment benefit = NNT."
            },
            "data": {"type": "nnt", "context": "Calculate NNT.", "r_treatment": 0.03, "r_control": 0.07,
                     "treatment_label": "Naloxone program", "control_label": "No program", "outcome_label": "overdose death"}
        },
        {
            "id": "adv_5", "title": "Scenario 5: Physical Activity & Hip Fracture",
            "description": "2,800 adults 65+ followed up to 10 years. Follow-up varies due to deaths and losses to follow-up. A Cox proportional hazards model was fitted.",
            "correct_measure": "Hazard Ratio (HR)",
            "measure_hint": "Three clues: (1) follow-up varies, (2) participants are censored, (3) **Cox model used**. The Cox model always produces a Hazard Ratio.",
            "measure_wrong": {
                "Population Attributable Risk (PAR)": "❌ PAR requires exposure prevalence and an RR. This study used Cox regression with censored follow-up — HR.",
                "Standardized Mortality Ratio (SMR)": "❌ SMR requires observed vs. expected from a reference population. Cox regression within a cohort produces HR.",
                "Attributable Risk & AR%": "❌ AR% requires complete fixed follow-up. Censored varying follow-up + Cox regression = HR.",
                "Number Needed to Harm / Treat (NNH/NNT)": "❌ NNT requires a fixed time point. Censored data and Cox regression = HR."
            },
            "data": {"type": "hr", "context": "Interpret the HR from the Cox model.",
                     "hr": 0.61, "ci_low": 0.48, "ci_high": 0.78,
                     "exposed_label": "Physically active", "outcome_label": "hip fracture"}
        },
        {
            "id": "adv_6", "title": "Scenario 6: PPI Use & Kidney Disease",
            "description": "3.2% of daily PPI users developed CKD over 5 years vs. 1.1% of non-users. How many patients need to take PPIs before one additional CKD case is expected?",
            "correct_measure": "Number Needed to Harm / Treat (NNH/NNT)",
            "measure_hint": "**'How many patients exposed before one additional harm'** = NNH. The most intuitive drug safety metric for clinicians. NNH = 1 / Risk Difference.",
            "measure_wrong": {
                "Population Attributable Risk (PAR)": "❌ PAR estimates population-level preventable fraction. This asks for per-patient harm — NNH.",
                "Standardized Mortality Ratio (SMR)": "❌ SMR compares to a reference population. Simple 5-year risk difference from a cohort = NNH.",
                "Attributable Risk & AR%": "❌ AR% tells the fraction attributable in exposed. The question asks for a per-patient clinical figure — NNH.",
                "Hazard Ratio (HR)": "❌ HR requires Cox regression with time-to-event. Cumulative 5-year risks + per-patient harm = NNH."
            },
            "data": {"type": "nnt", "context": "Calculate NNH for PPI use.", "r_treatment": 0.032, "r_control": 0.011,
                     "treatment_label": "Long-term PPI use", "control_label": "No PPI use", "outcome_label": "chronic kidney disease"}
        },
        {
            "id": "adv_7", "title": "Scenario 7: Vigorous Exercise & Cardiac Arrest (Case-Crossover)",
            "description": "A case-crossover study of 345 cardiac arrest survivors finds that vigorous exertion in the hour before arrest was 2.8 times more likely than during matched control periods (OR = 2.8). A cardiologist now asks: if 15% of the general population engages in regular vigorous exercise, what fraction of all cardiac arrests in the population could be attributed to this transient triggering effect?",
            "correct_measure": "Population Attributable Risk (PAR)",
            "measure_hint": "The question shifts from the individual OR to the **population-level preventable fraction** — what share of all cardiac arrests could be attributed to vigorous exertion as a trigger, given how common it is. PAR uses Pe (15% of population exercises vigorously) and RR ≈ OR (2.8) to answer this.",
            "measure_wrong": {
                "Standardized Mortality Ratio (SMR)": "❌ SMR compares observed to expected deaths vs. a reference population. This question asks what fraction of cardiac arrests in the general population is attributable to exercise — that's PAR.",
                "Attributable Risk & AR%": "❌ AR% estimates the fraction of disease within the **exposed group** (exercisers) attributable to exercise. The question asks about the fraction across the **total population** — that's PAR.",
                "Number Needed to Harm / Treat (NNH/NNT)": "❌ NNH asks how many people need exposure before one additional harm. The question asks what fraction of all cardiac arrests in the population are attributable to exercise — PAR.",
                "Hazard Ratio (HR)": "❌ HR compares event rates over time using Cox regression. This study produced an OR from a case-crossover design, and the question asks about population-level attributable fraction — PAR."
            },
            "data": {"type": "par", "context": "Calculate PAR% — fraction of all cardiac arrests attributable to vigorous exertion as a trigger.",
                     "Pe": 0.15, "RR": 2.8,
                     "Pe_label": "Prevalence of regular vigorous exercise in population",
                     "RR_label": "OR from case-crossover study (used as RR approximation)"}
        },
    ]

    measure_options = ["— Select —","Population Attributable Risk (PAR)","Standardized Mortality Ratio (SMR)",
                       "Attributable Risk & AR%","Number Needed to Harm / Treat (NNH/NNT)","Hazard Ratio (HR)"]

    if "adv_scenario_order" not in st.session_state:
        order = list(range(len(ADV_SCENARIOS))); random.shuffle(order)
        st.session_state["adv_scenario_order"] = order
    SHUFFLED_ADV = [ADV_SCENARIOS[i] for i in st.session_state["adv_scenario_order"]]

    col_hdr5, col_rst5 = st.columns([5,1])
    with col_hdr5: st.caption(f"**{len(ADV_SCENARIOS)} scenarios** — randomized. Reset to shuffle.")
    with col_rst5:
        if st.button("🔄 Reset", key="reset_tab5"):
            for sc in ADV_SCENARIOS:
                for k_suffix in ["measure","submitted"]:
                    k = f"adv_{k_suffix}_{sc['id']}"
                    if k in st.session_state: del st.session_state[k]
            if "adv_scenario_order" in st.session_state: del st.session_state["adv_scenario_order"]
            st.rerun()

    for sc in SHUFFLED_ADV:
        st.divider(); st.subheader(sc["title"]); st.markdown(sc["description"])
        sid = sc["id"]
        submitted_key = f"adv_submitted_{sid}"
        already_submitted = st.session_state.get(submitted_key, False)

        measure_choice = st.selectbox("Which advanced measure is most appropriate?",
            measure_options, key=f"adv_measure_{sid}", disabled=already_submitted)

        selected = st.session_state.get(f"adv_measure_{sid}") not in [None, "— Select —"]

        if not already_submitted:
            if selected:
                if st.button("Submit My Answer", key=f"adv_submit_{sid}", type="primary"):
                    st.session_state[submitted_key] = True; st.rerun()
            else:
                st.caption("⬆️ Select a measure before submitting.")

        if already_submitted:
            measure_val = st.session_state.get(f"adv_measure_{sid}")
            correct = measure_val == sc["correct_measure"]

            if not correct:
                wrong_hint = sc.get("measure_wrong",{}).get(measure_val,"")
                st.error("📋 **Incorrect — here's what you missed:**")
                if wrong_hint: st.markdown(f"**You selected:** *{measure_val}*\n\n{wrong_hint}")
                st.markdown(f"✅ **Correct:** {sc['correct_measure']} — {sc['measure_hint']}")
                if st.button("🔄 Try Again", key=f"adv_retry_{sid}"):
                    for k_suffix in ["measure","submitted"]:
                        k = f"adv_{k_suffix}_{sid}"
                        if k in st.session_state: del st.session_state[k]
                    st.rerun()
            else:
                st.success(f"✅ Correct! **{sc['correct_measure']}** — {sc['measure_hint']}")

            if correct:
                st.markdown("---"); st.markdown("### 📋 Now run the analysis")
                st.markdown(sc["data"]["context"]); d = sc["data"]
                if d["type"] == "par":
                    col1,col2 = st.columns(2)
                    col1.metric("Exposure Prevalence (Pe)", f"{round(d['Pe']*100,1)}%"); col2.metric("Risk Ratio (RR)", d["RR"])
                    if st.button("Calculate PAR%", key=f"run_{sid}"):
                        PAR_pct = (d["Pe"]*(d["RR"]-1))/(1+d["Pe"]*(d["RR"]-1))*100
                        st.metric("PAR%", f"{round(PAR_pct,1)}%")
                        st.success(f"{round(PAR_pct,1)}% of all cases attributable to this exposure.")
                elif d["type"] == "smr":
                    col1,col2 = st.columns(2); col1.metric("Observed", d["observed"]); col2.metric("Expected", d["expected"])
                    if st.button("Calculate SMR", key=f"run_{sid}"):
                        smr = d["observed"]/d["expected"]
                        ci_low_s = max(0, smr-1.96*(smr/math.sqrt(d["observed"]))); ci_high_s = smr+1.96*(smr/math.sqrt(d["observed"]))
                        st.metric("SMR", round(smr,3)); st.write(f"95% CI: ({round(ci_low_s,3)}, {round(ci_high_s,3)})")
                        if ci_low_s <= 1 <= ci_high_s: st.warning("CI includes 1 — not significantly different from reference.")
                        elif smr > 1: st.error(f"SMR = {round(smr,2)} — Excess mortality.")
                        else: st.success(f"SMR = {round(smr,2)} — Lower than expected. May reflect healthy worker effect.")
                        draw_ci("SMR", smr, ci_low_s, ci_high_s)
                elif d["type"] == "ar":
                    col1,col2 = st.columns(2)
                    col1.metric(f"Risk ({d['exposed_label']})", f"{round(d['r_exposed']*100,1)}%")
                    col2.metric(f"Risk ({d['unexposed_label']})", f"{round(d['r_unexposed']*100,1)}%")
                    if st.button("Calculate AR & AR%", key=f"run_{sid}"):
                        ar = d["r_exposed"]-d["r_unexposed"]; ar_pct = (ar/d["r_exposed"])*100
                        col1,col2 = st.columns(2); col1.metric("AR", f"{round(ar*100,1)}%"); col2.metric("AR%", f"{round(ar_pct,1)}%")
                        st.success(f"AR = {round(ar*100,1)}%: absolute excess risk."); st.success(f"AR% = {round(ar_pct,1)}%: fraction attributable in exposed group.")
                elif d["type"] == "nnt":
                    col1,col2 = st.columns(2)
                    col1.metric(f"Risk ({d['treatment_label']})", f"{round(d['r_treatment']*100,1)}%")
                    col2.metric(f"Risk ({d['control_label']})", f"{round(d['r_control']*100,1)}%")
                    if st.button("Calculate NNT/NNH", key=f"run_{sid}"):
                        risk_diff = abs(d["r_treatment"]-d["r_control"]); nnt = round(1/risk_diff,1)
                        st.metric("NNT/NNH", nnt)
                        if d["r_treatment"] < d["r_control"]: st.success(f"NNT = {nnt}: treat {nnt} to prevent one additional {d['outcome_label']}.")
                        else: st.error(f"NNH = {nnt}: expose {nnt} to cause one additional {d['outcome_label']}.")
                elif d["type"] == "hr":
                    col1,col2,col3 = st.columns(3); col1.metric("HR", d["hr"]); col2.metric("CI Lower", d["ci_low"]); col3.metric("CI Upper", d["ci_high"])
                    if st.button("Interpret HR", key=f"run_{sid}"):
                        if d["ci_low"] <= 1 <= d["ci_high"]: st.warning(f"HR = {d['hr']} — CI includes 1. Not significant.")
                        elif d["hr"] < 1: st.success(f"HR = {d['hr']}: {d['exposed_label']} had {round((1-d['hr'])*100,1)}% lower hazard of {d['outcome_label']}. Significant.")
                        else: st.error(f"HR = {d['hr']}: {d['exposed_label']} had {round((d['hr']-1)*100,1)}% higher hazard of {d['outcome_label']}. Significant.")
                        draw_ci("HR", d["hr"], d["ci_low"], d["ci_high"])

    st.divider()
    adv_answered = sum(1 for sc in ADV_SCENARIOS if st.session_state.get(f"adv_submitted_{sc['id']}"))
    adv_correct = sum(1 for sc in ADV_SCENARIOS if st.session_state.get(f"adv_submitted_{sc['id']}") and st.session_state.get(f"adv_measure_{sc['id']}") == sc["correct_measure"])
    if adv_answered > 0:
        st.subheader(f"📊 Score: {adv_correct} / {adv_answered} submitted")
        st.progress(adv_correct/adv_answered)

    st.markdown("---")
    st.markdown("Strong epidemiologists think structurally before computing.")

# ==================================================
# TAB 6: HYPOTHESIS TESTING
# QUICK WIN 2: One-tailed pop-up connects H0/H1 to tail choice
# ==================================================
with tab6:
    col_t6, col_r6 = st.columns([5,1])
    with col_r6:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_tab6"):
            for k in list(st.session_state.keys()):
                if any(k.startswith(p) for p in ["h0_","h1_","tails_","ht_section","chi2_slider","dof_select","tail_radio"]):
                    del st.session_state[k]
            st.rerun()
    st.markdown("Build your understanding of hypothesis testing — writing hypotheses correctly, understanding tails, and interpreting p-values.")

    ht_section = st.radio("Choose a section:", [
        "1️⃣ Hypothesis Builder",
        "2️⃣ One vs. Two Tailed Tests",
        "3️⃣ What Does Rejecting the Null Actually Mean?"
    ], horizontal=True, key="ht_section")
    st.divider()

    if ht_section == "1️⃣ Hypothesis Builder":
        st.subheader("Hypothesis Builder")
        with st.expander("📖 Quick Reference: Null vs. Alternative Hypothesis"):
            st.markdown("""
**H₀ (Null):** No association, no difference. Always written as an equality (RR = 1, μ₁ = μ₂). What you're trying to find evidence against.

**H₁ (Alternative):** States an association or effect exists.
- **Two-tailed (≠):** You're not predicting which direction — just that a difference exists. Your 5% error tolerance is split across both directions (2.5% each). Use this as the default when you have no strong prior reason to predict direction.
- **One-tailed (< or >):** You're predicting a specific direction before collecting data. All 5% of your error tolerance goes toward detecting an effect in that direction — making it more sensitive there, but blind to effects the other way. Only use when strong prior evidence supports the direction.

**Key principle:** You never *prove* H₀ true. You either reject it (p < 0.05) or fail to reject it (p ≥ 0.05).
            """)

        HYP_SCENARIOS = [
            {
                "id": "h1", "title": "Scenario A: Exercise & Blood Pressure",
                "description": "A researcher tests whether a 12-week aerobic program reduces systolic BP in hypertensive adults. She expects it to **decrease** BP based on prior research.",
                "null_options": ["The program has no effect on systolic BP (μ_before = μ_after)",
                                  "The program reduces BP (μ_before > μ_after)",
                                  "BP changes in either direction (μ_before ≠ μ_after)"],
                "alt_options": ["The program reduces systolic BP (μ_before > μ_after)",
                                 "The program has no effect",
                                 "BP changes in either direction (μ_before ≠ μ_after)"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "one-tailed",
                "null_feedback": "✅ Correct. H₀ states no effect — BP before = BP after.",
                "null_wrong_feedback": "❌ H₀ must state no effect — BP before = BP after.",
                "alt_feedback": "✅ Correct. Specific directional prediction (reduction) → one-tailed.",
                "alt_wrong_feedback": "❌ The researcher expects a decrease. That directional prediction makes this one-tailed, not two-tailed.",
                "tails_connection": "🎯 **Your H₁ specifies a direction** (μ_before > μ_after) — you predicted a reduction before collecting data. So this is a **one-tailed test**.\n\nIn plain terms: when we decide to reject H₀, we're willing to be wrong 5% of the time (that's α = 0.05). In a one-tailed test, you spend all of that 5% looking for an effect in the one direction you predicted. That makes you better at detecting an effect in that direction — but completely blind to effects in the opposite direction. If BP actually increased, this test could not detect it."
            },
            {
                "id": "h2", "title": "Scenario B: New Drug & Liver Enzymes",
                "description": "A company tests whether a new cholesterol drug changes liver enzyme elevation rates vs. placebo. **No prior evidence** about direction.",
                "null_options": ["Drug and placebo have the same enzyme elevation rate (p_drug = p_placebo)",
                                  "Drug increases enzyme elevation (p_drug > p_placebo)",
                                  "Drug changes enzyme levels (p_drug ≠ p_placebo)"],
                "alt_options": ["Drug causes enzyme elevation at a different rate (p_drug ≠ p_placebo)",
                                  "Drug increases enzyme elevation (p_drug > p_placebo)",
                                  "Drug has no effect"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "two-tailed",
                "null_feedback": "✅ Correct. H₀ states no difference — equal rates in both groups.",
                "null_wrong_feedback": "❌ H₀ must state no difference.",
                "alt_feedback": "✅ Correct. No directional prediction → two-tailed (≠).",
                "alt_wrong_feedback": "❌ No prior directional basis. Without a directional prediction, H₁ should be two-tailed (≠).",
                "tails_connection": "🎯 **Your H₁ uses ≠** — you're not predicting which direction the effect will go, just that a difference exists. So this is a **two-tailed test**.\n\nIn plain terms: you still have the same 5% tolerance for being wrong, but now you split it — 2.5% goes toward detecting an increase, and 2.5% goes toward detecting a decrease. You can catch an effect in either direction, but because you're dividing your sensitivity, you need slightly stronger evidence to call it significant compared to a one-tailed test. That's the tradeoff."
            },
            {
                "id": "h3", "title": "Scenario C: Vaccine & Respiratory Illness",
                "description": "Epidemiologists test whether a new vaccine reduces respiratory illness. Based on known mechanism, they expect it to be **protective**.",
                "null_options": ["Vaccine and unvaccinated groups have the same incidence (IRR = 1)",
                                  "Vaccine reduces incidence (IRR < 1)",
                                  "Vaccine changes incidence in either direction (IRR ≠ 1)"],
                "alt_options": ["Vaccine reduces incidence (IRR < 1)",
                                  "Vaccine changes incidence in either direction (IRR ≠ 1)",
                                  "Vaccine has no effect (IRR = 1)"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "one-tailed",
                "null_feedback": "✅ Correct. H₀: no effect — equal incidence (IRR = 1).",
                "null_wrong_feedback": "❌ H₀ must state no effect — IRR = 1.",
                "alt_feedback": "✅ Correct. Known protective mechanism → directional → one-tailed (IRR < 1).",
                "alt_wrong_feedback": "❌ Known protective mechanism supports a directional prediction — one-tailed.",
                "tails_connection": "🎯 **Your H₁ specifies a direction** (IRR < 1) — you predicted protection based on the vaccine's known mechanism. So this is a **one-tailed test**.\n\nIn plain terms: all of your 5% tolerance for error is focused on detecting a reduction in incidence. This makes the test more sensitive to finding a protective effect — but if the vaccine somehow increased incidence, this test would miss it entirely. One-tailed tests are only appropriate when strong prior evidence makes the direction clear before you collect data."
            },
            {
                "id": "h4", "title": "Scenario D: Screen Time & Obesity",
                "description": "A chi-square test examines whether obesity prevalence differs between high/low screen time groups. **No prior directional hypothesis.**",
                "null_options": ["No association between screen time and obesity (PR = 1 / independent)",
                                  "High screen time increases obesity (PR > 1)",
                                  "Association exists (PR ≠ 1)"],
                "alt_options": ["There is an association (PR ≠ 1)",
                                  "High screen time increases obesity (PR > 1)",
                                  "No association"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "two-tailed",
                "null_feedback": "✅ Correct. H₀: no association — independence.",
                "null_wrong_feedback": "❌ H₀ must state no association.",
                "alt_feedback": "✅ Correct. No directional prediction + chi-square is always two-tailed.",
                "alt_wrong_feedback": "❌ No directional prediction, and chi-square tests are **always two-tailed**.",
                "tails_connection": "🎯 **Your H₁ uses ≠** — no direction predicted. So this is a **two-tailed test**. And there's an important additional rule here: **chi-square tests are always two-tailed**, regardless of how the hypotheses are written.\n\nIn plain terms: a two-tailed test splits the 5% tolerance for error in both directions — 2.5% each way — so you can detect an association regardless of which group has higher prevalence. Chi-square tests are designed this way because they measure discrepancy in a table without regard to direction. You can't make a chi-square one-tailed."
            },
        ]

        for sc in HYP_SCENARIOS:
            st.divider(); st.markdown(f"**{sc['title']}**"); st.markdown(sc["description"])
            sid = sc["id"]
            st.markdown("**Step 1: Select the correct null hypothesis (H₀):**")
            null_choice = st.radio("H₀:", sc["null_options"], key=f"h0_{sid}", index=None, label_visibility="collapsed")
            if null_choice is not None:
                if sc["null_options"].index(null_choice) == sc["correct_null_idx"]: st.success(sc["null_feedback"])
                else: st.error(sc["null_wrong_feedback"])

            st.markdown("**Step 2: Select the correct alternative hypothesis (H₁):**")
            alt_choice = st.radio("H₁:", sc["alt_options"], key=f"h1_{sid}", index=None, label_visibility="collapsed")
            if alt_choice is not None:
                if sc["alt_options"].index(alt_choice) == sc["correct_alt_idx"]: st.success(sc["alt_feedback"])
                else: st.error(sc["alt_wrong_feedback"])

            # QUICK WIN: Only show Step 3 (tails) after both H0 and H1 are correct
            # AND show the connecting explanation first
            if null_choice is not None and alt_choice is not None:
                null_correct = sc["null_options"].index(null_choice) == sc["correct_null_idx"]
                alt_correct = sc["alt_options"].index(alt_choice) == sc["correct_alt_idx"]
                if null_correct and alt_correct:
                    st.info(sc["tails_connection"])
                    st.markdown("**Step 3: Based on your hypotheses — is this a one-tailed or two-tailed test?**")
                    tail_answer = st.radio("Tails:", ["One-tailed","Two-tailed"], key=f"tails_{sid}", index=None, horizontal=True, label_visibility="collapsed")
                    if tail_answer is not None:
                        expected = "One-tailed" if sc["correct_tails"] == "one-tailed" else "Two-tailed"
                        if tail_answer == expected:
                            reason = "Your H₁ specifies a direction (< or >), so all of your error tolerance goes toward detecting an effect in that one direction." if sc["correct_tails"] == "one-tailed" else "Your H₁ uses ≠ — no direction predicted — so you split your error tolerance equally across both directions."
                            st.success(f"✅ Correct — **{tail_answer.lower()}** test. {reason}")
                        else:
                            if sc["correct_tails"] == "one-tailed":
                                st.error("❌ This should be **one-tailed**. Look at your H₁ — it predicts a specific direction (< or >). When you predict a direction before collecting data, you focus all of your 5% error tolerance on detecting an effect that way. That's a one-tailed test. The tradeoff: you can't detect effects in the opposite direction.")
                            else:
                                st.error("❌ This should be **two-tailed**. Look at your H₁ — it uses ≠, meaning you're not predicting a direction. When you don't predict a direction, you split your 5% error tolerance — 2.5% for detecting an increase, 2.5% for detecting a decrease. That's a two-tailed test. It's the default in epidemiology.")

    elif ht_section == "2️⃣ One vs. Two Tailed Tests":
        st.subheader("One vs. Two Tailed Tests — Interactive Visualization")
        col1, col2 = st.columns([3,2])

        with col2:
            chi2_input = st.slider("Chi-square statistic (χ²)", min_value=0.0, max_value=15.0, value=3.84, step=0.01)
            dof_input = st.selectbox("Degrees of freedom", [1,2,3,4,5], index=0)
            tail_choice = st.radio("Tail type:", ["Two-tailed (÷2 on each side)","One-tailed (all on one side)"], key="tail_radio")

            from scipy.stats import chi2 as chi2_dist
            p_two = float(1 - chi2_dist.cdf(chi2_input, dof_input))
            p_one = p_two / 2
            p_display = p_two if "Two-tailed" in tail_choice else p_one
            tail_label = "Two-tailed p" if "Two-tailed" in tail_choice else "One-tailed p"
            st.metric(tail_label, f"{round(p_display,4)}" if p_display >= 0.0001 else "< 0.0001")
            if p_display < 0.05: st.success("p < 0.05 → Reject H₀")
            else: st.warning("p ≥ 0.05 → Fail to reject H₀")
            st.markdown(f"**Two-tailed p:** {round(p_two,4)}  |  **One-tailed p:** {round(p_one,4)}")
            st.caption("One-tailed p is always exactly half the two-tailed p.")

            # QUICK WIN: One-tailed guidance pop-up
            st.markdown("---")
            with st.expander("💡 When is one-tailed appropriate?"):
                st.markdown("""
**The core idea:** Every test has a 5% tolerance for being wrong when rejecting H₀ (α = 0.05). The question is how you spend it.

- **Two-tailed:** Split the 5% equally — 2.5% watching for an increase, 2.5% watching for a decrease. You can detect effects in either direction.
- **One-tailed:** Put all 5% in one direction. More sensitive to finding an effect there — but completely unable to detect an effect the other way.

**Use one-tailed when ALL three are true:**
1. You have a **specific directional hypothesis** established *before* seeing the data
2. You have **strong prior evidence** supporting that direction
3. An effect in the opposite direction would be **clinically meaningless**

**Use two-tailed (the default) when:**
- You're testing whether *any* difference exists
- You're unsure which direction the effect will go
- You're using a **chi-square test** (always two-tailed — it can't be made one-tailed)
- You decided on direction *after* seeing your data — that's p-hacking

⚠️ **Warning:** Switching from two-tailed to one-tailed after seeing p = 0.07 to make it "significant" (p = 0.035) is data manipulation. The tail choice must be made before data collection.
                """)

        with col1:
            from scipy.stats import chi2 as chi2_dist2
            x_vals = [i*0.1 for i in range(0,160)]
            y_vals = [float(chi2_dist2.pdf(x,dof_input)) if x > 0 else 0 for x in x_vals]
            max_y = max(y_vals) if max(y_vals) > 0 else 1
            w,h = 520,220; ml,mr,mb,mt = 40,20,40,20
            pw,ph = w-ml-mr, h-mb-mt; x_max_plot=15.0

            def px(xv): return ml + (xv/x_max_plot)*pw
            def py(yv): return mt + ph - (yv/(max_y*1.1))*ph

            path_pts=[]
            for i,(xv,yv) in enumerate(zip(x_vals,y_vals)):
                if xv > x_max_plot: break
                path_pts.append(f"{'M' if i==0 else 'L'}{round(px(xv),1)},{round(py(yv),1)}")
            curve_path=" ".join(path_pts)
            rc="#c0392b"
            fill_pts=[f"{round(px(xv),1)},{round(py(yv),1)}" for xv,yv in zip(x_vals,y_vals) if chi2_input <= xv <= x_max_plot]
            if fill_pts:
                fill_path=f"M{round(px(chi2_input),1)},{round(py(0),1)} "+" ".join(f"L{pt}" for pt in fill_pts)+f" L{round(px(x_max_plot),1)},{round(py(0),1)} Z"
            else: fill_path=""
            ticks=[0,2,4,6,8,10,12,14]
            tick_svg="".join(f'<text x="{round(px(xt),1)}" y="{h-8}" text-anchor="middle" font-size="11" fill="#555">{xt}</text><line x1="{round(px(xt),1)}" y1="{h-mb}" x2="{round(px(xt),1)}" y2="{h-mb+4}" stroke="#555" stroke-width="1"/>' for xt in ticks)
            crit_line=f'<line x1="{round(px(chi2_input),1)}" y1="{mt}" x2="{round(px(chi2_input),1)}" y2="{h-mb}" stroke="{rc}" stroke-width="2" stroke-dasharray="5,3"/>'
            lbl=f"χ²={round(chi2_input,2)}"; lw=max(len(lbl)*7+12,70)
            lx=round(min(px(chi2_input)+10,w-mr-lw-4),1); ly=round((mt+h-mb)/2,1)
            crit_label=f'<rect x="{lx}" y="{ly-14}" width="{lw}" height="19" fill="white" stroke="{rc}" stroke-width="1.5" rx="3"/><text x="{lx+lw//2}" y="{ly}" text-anchor="middle" font-size="12" fill="{rc}" font-weight="bold">{lbl}</text>'
            p_ann=f'<text x="{round(min(px(chi2_input)+1.5*pw/15,w-mr-30),1)}" y="{mt+30}" text-anchor="middle" font-size="11" fill="{rc}">p={round(p_display,4) if p_display>=0.0001 else "<0.0001"}</text>'
            axes=f'<line x1="{ml}" y1="{h-mb}" x2="{w-mr}" y2="{h-mb}" stroke="#333" stroke-width="1.5"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{h-mb}" stroke="#333" stroke-width="1.5"/>'
            labels=f'<text x="{w//2}" y="{h-2}" text-anchor="middle" font-size="12" fill="#333">χ² value</text><text x="{w//2}" y="{mt-8}" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">χ² Distribution (df={dof_input})</text>'
            fill_svg_part = f'<path d="{fill_path}" fill="{rc}" opacity="0.4"/>' if fill_path else ""
            svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="font-family:sans-serif;background:#f9f9f9;border-radius:8px;">{axes}{fill_svg_part}<path d="{curve_path}" fill="none" stroke="#2c3e50" stroke-width="2.5"/>{crit_line}{crit_label}{p_ann}{tick_svg}{labels}</svg>'
            st.markdown(svg, unsafe_allow_html=True)

        st.markdown("""
**Two-tailed (default):** You're not predicting a direction — just asking whether a difference exists. Your 5% error tolerance is split: 2.5% goes toward detecting an increase, 2.5% toward detecting a decrease. You can catch effects in either direction. Chi-square is always two-tailed.

**One-tailed:** You've predicted a specific direction before collecting data. All 5% of your error tolerance goes toward detecting an effect in that one direction, making the test more sensitive there — but completely unable to detect an effect the other way. Only use when strong prior evidence justifies the directional prediction.
        """)

    elif ht_section == "3️⃣ What Does Rejecting the Null Actually Mean?":
        st.subheader("What Does Rejecting the Null Actually Mean?")
        with st.expander("🔵 What the p-value IS", expanded=True):
            st.markdown("""
**The p-value is the probability of observing a result as extreme as yours (or more extreme) if the null hypothesis were true.**

Small p (e.g., 0.003): data this extreme would occur only 0.3% of the time under H₀ — very surprising.
Large p (e.g., 0.42): data this extreme would occur 42% of the time under H₀ — not surprising.

The 0.05 threshold means we accept a 5% chance of rejecting a true H₀ (Type I error / false positive).
            """)
        with st.expander("🔴 What the p-value is NOT"):
            st.markdown("""
| ❌ Common Misconception | ✅ What's Actually True |
|---|---|
| "p = 0.03 means 3% chance H₀ is true" | p-value says nothing about the probability H₀ is true |
| "p = 0.06 means no association" | Failing to reject H₀ does not prove no effect |
| "p < 0.05 means the result is important" | Statistical significance ≠ practical significance |
| "We accept the null hypothesis" | You never *accept* H₀ — you fail to reject it |
| "p = 0.049 is meaningful, p = 0.051 is not" | The 0.05 cutoff is arbitrary |
| "Smaller p = stronger association" | p reflects both sample size AND effect size |
            """)
        with st.expander("🟡 Type I and Type II Errors"):
            st.markdown("""
|  | H₀ is TRUE | H₀ is FALSE |
|---|---|---|
| **Reject H₀** | ❌ Type I Error (α) | ✅ Correct (Power = 1−β) |
| **Fail to reject H₀** | ✅ Correct | ❌ Type II Error (β) |

**Type I (α):** False positive — rejecting a true H₀. Probability = α (0.05).
**Type II (β):** False negative — failing to reject a false H₀.
**Tradeoff:** Decreasing α reduces Type I errors but increases Type II without a larger sample size.
            """)
        with st.expander("🟢 CI Connection"):
            st.markdown("""
**95% CI and p-value always agree:**
- CI includes 1 → p ≥ 0.05 → fail to reject H₀
- CI excludes 1 → p < 0.05 → reject H₀

The CI gives more information — it shows the range of plausible effect sizes, not just whether to reject H₀.
            """)

    st.markdown("---")
    st.markdown("Strong epidemiologists think structurally before computing.")

# ==================================================
# TAB 7: GLOSSARY (QUICK WIN 3 — new tab)
# ==================================================
with tab7:
    st.subheader("📖 Glossary of Key Terms")
    st.markdown("Use this as a reference while working through practice scenarios or analyzing data.")

    with st.expander("📐 Study Designs", expanded=True):

        st.markdown("### ❓ How does the study start?")
        st.markdown("---")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("#### 🟩 Cohort Study")
            st.markdown("**Starts with:** Exposure status")
            st.markdown("**Logic:** Grouped by exposure → outcome ascertained")
            st.markdown("**Prospective:** Data collected going forward in time")
            st.markdown("**Retrospective:** Historical records used; exposure still defined before outcome")
            st.markdown("")
            st.markdown("**Timeline:**")
            st.markdown("```\n① Exposure defined\n        ↓\n② Outcome measured\n```")
            st.success("Produces: **RR / IRR**")
            st.markdown("*Key question: Who develops the outcome among exposed vs. unexposed?*")

        with col_b:
            st.markdown("#### 🟦 Case-Control Study")
            st.markdown("**Starts with:** Outcome status")
            st.markdown("**Logic:** Recruit cases + controls → look backward at past exposure")
            st.markdown("**Always:** Uses existing disease status; cannot be prospective")
            st.markdown("**Matched variant:** Cases and controls paired on confounders (age, sex). Same OR logic; controls confounding by design.")
            st.markdown("**Timeline:**")
            st.markdown("```\n① Past exposure (recalled)\n        ↑\n② Start here: disease yes/no\n```")
            st.info("Produces: **OR**")
            st.markdown("*Key question: Were cases more likely to have been exposed than controls?*")

        with col_c:
            st.markdown("#### 🟧 Cross-Sectional Study")
            st.markdown("**Starts with:** A sample of people")
            st.markdown("**Logic:** Exposure and outcome measured at the same moment")
            st.markdown("**Always:** One point in time — a snapshot. Cannot establish temporal order")
            st.markdown("")
            st.markdown("")
            st.markdown("**Timeline:**")
            st.markdown("```\nExposure ─┐\n          ├─ measured simultaneously\nOutcome  ─┘\n```")
            st.warning("Produces: **PR**")
            st.markdown("*Key question: Is exposure associated with current disease prevalence?*")

        st.markdown("---")
        st.markdown("#### 🟪 Advanced Design: Case-Crossover Study")
        st.markdown("A case-crossover study is a variant of the case-control design where **each case serves as their own control** — exposure during a hazard period immediately before the event is compared to exposure during a control period for the same person.")

        ccol1, ccol2, ccol3 = st.columns(3)
        with ccol1:
            st.markdown("**Starts with:** Cases only (people who had the event)")
            st.markdown("**Logic:** Compare each person's exposure at the time of the event vs. at a control time when no event occurred")
        with ccol2:
            st.markdown("**Timeline:**")
            st.markdown("```\nControl period  →  Hazard period\n(same person,      (just before\n no event)          the event)\n```")
            st.markdown("*Eliminates between-person confounding — each person is their own control*")
        with ccol3:
            st.markdown("**Best for:** Transient exposures with acute effects (e.g., air pollution spike → MI, alcohol → injury)")
            st.markdown("**Not for:** Chronic, stable exposures where 'hazard period' concept doesn't apply")
            st.markdown("Produces: **OR**")

        st.markdown("*Key question: Was the person more exposed just before the event than during a typical period?*")

        st.markdown("---")
        st.markdown("""
**Cohort Study**
Participants are classified by **exposure status**, and the study follows the logic of exposure → outcome. Can be **prospective** (data collected going forward) or **retrospective** (historical records used, but exposure still defined before outcome). The defining feature is grouping by exposure, not when data were collected. Produces RR or IRR.

**Case-Control Study**
Participants recruited by **outcome status** — cases (have disease) and controls (don't). Researchers look **backward** at past exposure. Produces OR. Efficient for rare diseases.

**Matched Case-Control:** A variant where each case is paired with one or more controls matched on potential confounders (e.g., age, sex, neighborhood). Matching controls confounding by design rather than analysis. The same OR logic applies, but analysis must account for the matched structure (conditional logistic regression).

**Cross-Sectional Study**
Exposure and outcome measured **at the same point in time** — a snapshot. Produces PR. Cannot establish temporal order.

**Case-Crossover Study**
Each case serves as **their own control** — the person's exposure just before their event (hazard period) is compared to their exposure at a matched control time when no event occurred. Eliminates between-person confounding on stable characteristics (age, sex, chronic health status) because the same person is compared to themselves. Best suited to **transient, short-acting exposures** with acute effects. Examples: air pollution spike and myocardial infarction, alcohol consumption and injury, vigorous exercise and cardiac arrest. Produces OR. Not appropriate when the exposure itself is stable or chronic, because there would be no meaningful contrast between hazard and control periods.

**RCT**
Participants **randomly assigned** to treatment or control. Gold standard for causation.
        """)

    with st.expander("📊 Variable Types"):
        st.markdown("""
**Binary Variable**
Exactly **two categories** — disease yes/no, vaccinated yes/no. Produces 2×2 table. Enables RR and OR. When there are only two options, you can directly compare exposed vs. unexposed with a single ratio.

**Why not categorical?** Binary is a special case — with only two categories you get the full suite of measures. Three or more categories → chi-square only.

**Categorical Variable (Nominal, >2 levels)**
**3 or more unordered categories** — blood type, disease severity. Chi-square only. No RR or OR.

**Why not binary?** More than two unordered categories means no single exposed vs. unexposed comparison. Chi-square tests independence across all cells.

**Ordinal Variable**
**Ordered categories** — pain scale, satisfaction. Treated like categorical here (chi-square).

**Rate Variable (Person-Time)**
**Counts per unit of observation time** when follow-up varies. Use when participants contribute different amounts of time at risk. Produces IRR.

**Why not binary?** If some are followed 6 months and others 2 years, simply counting cases is unfair. Person-time standardizes for varying observation time.
        """)

    with st.expander("🔢 Measures of Association"):
        st.markdown("""
**Risk Ratio (RR)** — Risk in exposed ÷ risk in unexposed. Cohort studies. RR = 1: no difference; RR > 1: higher risk; RR < 1: protective.

**Prevalence Ratio (PR)** — Same formula as RR, but used in cross-sectional studies where the outcome is already existing (prevalent), not new (incident).

**Odds Ratio (OR)** — Odds of outcome in exposed ÷ odds in unexposed. Used in case-control studies. OR is always farther from 1 than RR when outcome is common. When outcome is rare (<10%), OR ≈ RR.

**Incidence Rate Ratio (IRR)** — Rate in exposed ÷ rate in unexposed, where rates use person-time denominators. Used when follow-up time varies.

**Hazard Ratio (HR)** — Ratio of instantaneous event rates at any moment in time. Output of Cox proportional hazards regression. Used when follow-up varies and participants may be censored.
        """)

    with st.expander("📉 Advanced Epi Measures"):
        st.markdown("""
**Attributable Risk (AR) / Risk Difference**
Risk in exposed − risk in unexposed. Absolute excess risk. Example: AR = 8% means 8 additional cases per 100 exposed vs. unexposed.

**Attributable Risk Percent (AR%)**
AR ÷ risk in exposed × 100. Fraction of disease **in the exposed group** attributable to exposure.

**Population Attributable Risk Percent (PAR%)**
Fraction of all disease **in the total population** attributable to an exposure. Formula: Pe × (RR − 1) / [1 + Pe × (RR − 1)] × 100. Accounts for both exposure prevalence and strength of association.

**Standardized Mortality Ratio (SMR)**
Observed deaths ÷ Expected deaths (expected calculated by applying reference population rates to the study group's age structure). SMR > 1: excess mortality. SMR < 1: lower mortality.

**Healthy Worker Effect**
Workers are generally healthier than the general population because very ill people cannot work. Can cause SMR < 1 in occupational cohorts even without a true protective effect.

**Number Needed to Treat (NNT)**
How many people need treatment for one additional person to benefit. NNT = 1 / Risk Difference.

**Number Needed to Harm (NNH)**
How many people need exposure for one additional person to be harmed. NNH = 1 / Risk Difference.
        """)

    with st.expander("🧪 Hypothesis Testing"):
        st.markdown("""
**Null Hypothesis (H₀)** — Default: no association, no difference. Always an equality (RR = 1, μ₁ = μ₂).

**Alternative Hypothesis (H₁)** — States an association exists. Two-tailed (≠) or one-tailed (< or >).

**p-value** — Probability of observing a result as extreme as yours if H₀ were true. NOT the probability that H₀ is true.

**One-tailed test** — Tests effect in one specific direction. All 5% in one tail. Only appropriate with strong prior directional hypothesis established before data collection.

**Two-tailed test** — Tests any difference regardless of direction. 5% split: 2.5% each tail. Default in epidemiology. Chi-square tests are always two-tailed.

**Type I Error (α)** — Rejecting true H₀. False positive. Probability = 0.05.

**Type II Error (β)** — Failing to reject false H₀. False negative. Power = 1 − β.

**Confidence Interval (CI)** — Range of plausible values for the true effect. 95% CI excluding 1 corresponds to p < 0.05. Gives more information than p-value alone — shows magnitude and precision.

**Chi-Square (χ²)** — Tests whether observed cell counts differ from expected if no association. Always two-tailed. Larger χ² = smaller p.

**Degrees of Freedom (df)** — For contingency table: (rows − 1) × (columns − 1). Affects chi-square distribution shape.
        """)

    with st.expander("📏 Standardization"):
        st.markdown("""
**Crude Rate** — Overall rate without adjusting for confounders. Can mislead when comparing populations with different age structures.

**Direct Standardization** — Applies each population's age-specific rates to a single standard population. Produces age-adjusted rate (per 100,000). Best for comparing two populations.

**Indirect Standardization** — Applies reference population rates to the study population's age structure. Produces SMR. Best when age-specific rates in your study population are unstable (small numbers).

**Confounding by Age** — Apparent difference in rates actually due to different age structures, not disease burden. Standardization removes this.
        """)

    st.markdown("---")
    st.markdown("*Return to any tab to apply these concepts in context.*")
