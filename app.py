import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------
# 1) Sayfa ayarları (geniş layout + başlık)
# ---------------------------------------------------------------
st.set_page_config(page_title="Titanic Passenger Insights", page_icon="🚢", layout="wide")

NAVY = "#1F3A5F"
TEAL = "#2A9D8F"
SURVIVAL_COLORS = {"Died": NAVY, "Survived": TEAL}
CLASS_COLORS = {"1st": TEAL, "2nd": "#5FA8D3", "3rd": NAVY}
SEX_COLORS = {"female": TEAL, "male": NAVY}
AGE_ORDER = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

st.markdown(
    f"""
    <style>
    h1, h2, h3 {{ color: {NAVY}; }}
    div[data-testid="stMetric"] {{
        background-color: #F4F8FB;
        border-left: 5px solid {TEAL};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# 2) Veriyi yükle (önce yerel dosya, yoksa URL)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("titanic.csv")
    except FileNotFoundError:
        return pd.read_csv(DATA_URL)


raw = load_data()                      # ham veri: karşılaştırma için hiç değiştirmiyoruz
cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
df = raw.loc[:, cols].copy()           # .loc ile kullanacağımız sütunları seçiyoruz

# ---------------------------------------------------------------
# 3) Eksik değerleri doldur
# ---------------------------------------------------------------
missing_before = df[["age", "embarked"]].isna().sum()
age_median = df["age"].median()
embarked_mode = df["embarked"].mode()[0]

df["age"] = df["age"].fillna(age_median)
df["embarked"] = df["embarked"].fillna(embarked_mode)
missing_after = df[["age", "embarked"]].isna().sum()

# ---------------------------------------------------------------
# 4) Min-Max normalizasyon (orijinal sütunlar korunuyor)
# ---------------------------------------------------------------
for c in ["age", "fare"]:
    df[f"{c}_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

# ---------------------------------------------------------------
# 5) Ayrıklaştırma: age -> age_group (orijinal ölçekteki yaş)
# ---------------------------------------------------------------
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, float("inf")],
    labels=AGE_ORDER,
    include_lowest=True,
)

# ---------------------------------------------------------------
# 6) Dummy coding (k-1): sex ve embarked, ilk kategori referans (survived'a dokunmuyoruz)
# ---------------------------------------------------------------
dummies = pd.get_dummies(df[["sex", "embarked"]], prefix=["sex", "embarked"], drop_first=True, dtype=int)
# embarked için effect coding (0, 1, -1): referans kategori C, tüm embarked sütunlarında -1 alır
dummies.loc[df["embarked"] == "C", ["embarked_Q", "embarked_S"]] = -1
transformed = pd.concat([df, dummies], axis=1)

# Grafikler için okunabilir etiketler
transformed["class_label"] = transformed["pclass"].map({1: "1st", 2: "2nd", 3: "3rd"})
transformed["outcome"] = transformed["survived"].map({0: "Died", 1: "Survived"})

# ---------------------------------------------------------------
# 7) Sidebar filtreleri
# ---------------------------------------------------------------
st.sidebar.header("Filters")
class_choice = st.sidebar.selectbox("Passenger class", ["All", "1st", "2nd", "3rd"])
sex_choice = st.sidebar.selectbox("Sex", ["All", "female", "male"])
age_choice = st.sidebar.selectbox("Age group", ["All"] + AGE_ORDER)
st.sidebar.caption("All metrics and charts update with these filters.")

filtered = transformed.copy()
if class_choice != "All":
    filtered = filtered[filtered["class_label"] == class_choice]
if sex_choice != "All":
    filtered = filtered[filtered["sex"] == sex_choice]
if age_choice != "All":
    filtered = filtered[filtered["age_group"] == age_choice]

is_empty = filtered.empty

# ---------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------
st.title("🚢 Titanic Passenger Insights")
st.markdown("Explore survival patterns across passenger groups.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Explore", "Preprocessing", "Data Explorer"])


def survival_rate_chart(data, group_col, label, colors, order):
    rate = data.groupby(group_col, observed=True)["survived"].mean().reindex(order).dropna().reset_index()
    rate["survived"] = rate["survived"] * 100
    fig = px.bar(
        rate, x=group_col, y="survived", color=group_col, color_discrete_map=colors,
        text=rate["survived"].round(1).astype(str) + "%",
        title=f"Survival rate by {label}",
        labels={group_col: label.capitalize(), "survived": "Survival rate (%)"},
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 100])
    return fig


# ---------------------------------------------------------------
# TAB 1 — Overview
# ---------------------------------------------------------------
with tab1:
    if is_empty:
        st.warning("No passengers match these filters. Try a different combination.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Passengers in selection", f"{len(filtered):,}")
        k2.metric("Survivors", f"{int(filtered['survived'].sum()):,}")
        k3.metric("Survival rate", f"{filtered['survived'].mean() * 100:.1f}%")
        k4.metric("Average fare", f"£{filtered['fare'].mean():.2f}")

        c1, c2 = st.columns(2)
        counts = filtered["outcome"].value_counts().reindex(["Died", "Survived"], fill_value=0).reset_index()
        counts.columns = ["outcome", "count"]
        fig_counts = px.bar(
            counts, x="outcome", y="count", color="outcome", color_discrete_map=SURVIVAL_COLORS,
            text="count", title="Survival counts",
            labels={"outcome": "Outcome", "count": "Number of passengers"},
        )
        fig_counts.update_layout(showlegend=False)
        c1.plotly_chart(fig_counts, width="stretch")
        c2.plotly_chart(
            survival_rate_chart(filtered, "class_label", "passenger class", CLASS_COLORS, ["1st", "2nd", "3rd"]),
            width="stretch",
            key="class_rate_overview",
        )
        st.caption("Note: rates describe this dataset only (891 passengers), not all Titanic passengers.")

# ---------------------------------------------------------------
# TAB 2 — Explore
# ---------------------------------------------------------------
with tab2:
    st.subheader("Compare passenger groups")
    if is_empty:
        st.warning("No passengers match these filters. Try a different combination.")
    else:
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            survival_rate_chart(filtered, "class_label", "passenger class", CLASS_COLORS, ["1st", "2nd", "3rd"]),
            width="stretch",
            key="class_rate_explore",
        )
        c2.plotly_chart(
            survival_rate_chart(filtered, "sex", "sex", SEX_COLORS, ["female", "male"]),
            width="stretch",
        )

        c3, c4 = st.columns(2)
        fig_age = px.histogram(
            filtered, x="age", color="outcome", color_discrete_map=SURVIVAL_COLORS, nbins=30,
            barmode="overlay", opacity=0.75, title="Age distribution by outcome",
            labels={"age": "Age (years)", "outcome": "Outcome", "count": "Number of passengers"},
        )
        fig_age.update_layout(yaxis_title="Number of passengers")
        c3.plotly_chart(fig_age, width="stretch")

        fig_fare = px.box(
            filtered, x="class_label", y="fare", color="class_label", color_discrete_map=CLASS_COLORS,
            category_orders={"class_label": ["1st", "2nd", "3rd"]}, title="Fare by passenger class",
            labels={"class_label": "Passenger class", "fare": "Fare (£)"},
        )
        fig_fare.update_layout(showlegend=False)
        c4.plotly_chart(fig_fare, width="stretch")

    # Bulgu tüm veriye göre hesaplanıyor (filtreden bağımsız)
    female_rate = transformed.loc[transformed["sex"] == "female", "survived"].mean() * 100
    male_rate = transformed.loc[transformed["sex"] == "male", "survived"].mean() * 100
    first_rate = transformed.loc[transformed["pclass"] == 1, "survived"].mean() * 100
    third_rate = transformed.loc[transformed["pclass"] == 3, "survived"].mean() * 100
    st.info(
        f"**Finding:** In this dataset, female passengers had a much higher survival rate "
        f"({female_rate:.1f}%) than male passengers ({male_rate:.1f}%), and first-class passengers "
        f"survived more often ({first_rate:.1f}%) than third-class passengers ({third_rate:.1f}%). "
        f"This is an observed association, not proof that sex or class caused survival."
    )

# ---------------------------------------------------------------
# TAB 3 — Preprocessing
# ---------------------------------------------------------------
with tab3:
    st.subheader("Data preparation steps")

    with st.expander("0. Load and inspect the raw data", expanded=True):
        r, c = raw.shape
        m1, m2 = st.columns(2)
        m1.metric("Rows", r)
        m2.metric("Columns", c)
        st.write("**Column names and data types**")
        st.dataframe(
            pd.DataFrame({"column": raw.columns, "dtype": raw.dtypes.astype(str).values,
                          "missing": raw.isna().sum().values}),
            hide_index=True, width="stretch",
        )
        st.write("**First five rows** (selected with `.iloc[:5]`)")
        st.dataframe(raw.iloc[:5], width="stretch")
        st.caption("The deck column is left out of the analysis because most of its values are missing.")

    with st.expander("1. Missing values"):
        st.write(
            f"Missing **age** values were filled with the median ({age_median:.1f}); missing **embarked** "
            f"values were filled with the most frequent port ('{embarked_mode}')."
        )
        st.dataframe(
            pd.DataFrame({"Missing before": missing_before, "Missing after": missing_after}),
            width="stretch",
        )

    with st.expander("2. Min-Max normalization"):
        st.write(
            "Min-Max normalization rescales a column to the 0–1 range: "
            "(x − min) / (max − min). The original age and fare columns are kept."
        )
        st.dataframe(df[["age", "age_norm", "fare", "fare_norm"]].head(10).round(3), width="stretch")

    with st.expander("3. Age discretization"):
        st.write(
            "Age is converted into groups: 0–12 Child, 13–18 Teen, 19–35 Young Adult, "
            "36–60 Adult, 61+ Older Adult."
        )
        a1, a2 = st.columns(2)
        a1.dataframe(df[["age", "age_group"]].head(10), width="stretch")
        a2.dataframe(
            df["age_group"].value_counts().reindex(AGE_ORDER).rename("passengers"),
            width="stretch",
        )

    with st.expander("4. Dummy coding"):
        st.write(
            "Dummy coding turns categories into numeric columns, keeping k−1 columns per variable. "
            "sex uses 0/1 coding with female as the reference (sex_male = 1 for male). "
            "embarked uses effect coding with values 0, 1 and −1: Q → (1, 0), S → (0, 1), and the "
            "reference port C (Cherbourg) → (−1, −1). "
            "survived is already 0/1, so it is not dummy-coded, and dummy columns are not normalized."
        )
        d1, d2 = st.columns(2)
        d1.write("**Before**")
        d1.dataframe(df[["sex", "embarked"]].head(), width="stretch")
        d2.write("**After**")
        d2.dataframe(dummies.head(), width="stretch")

# ---------------------------------------------------------------
# TAB 4 — Data Explorer
# ---------------------------------------------------------------
with tab4:
    st.subheader("Data Explorer")
    view = st.radio("Select view", ["Raw", "Cleaned", "Transformed"], horizontal=True)

    idx = filtered.index  # filtreler tüm görünümlere uygulanıyor
    if view == "Raw":
        table = raw.loc[idx]
        default_cols = cols
    elif view == "Cleaned":
        table = df.loc[idx, cols + ["age_group"]]
        default_cols = cols
    else:
        table = transformed.loc[idx].drop(columns=["class_label", "outcome"])
        default_cols = ["survived", "pclass", "age", "age_norm", "fare", "fare_norm", "age_group"] + list(dummies.columns)

    chosen = st.multiselect("Choose columns to display", list(table.columns), default=default_cols)
    st.write(f"**Rows after filters:** {len(table):,}")
    st.dataframe(table[chosen] if chosen else table, width="stretch", height=400)
