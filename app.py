from datetime import date
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.data_validation import validate_dataset
from src.preprocessing import prepare_features
from src.recommendations import make_recommendation
from src.predict import predict_attendance, MODEL_PATH
from src.train import DATA_PATH, METRICS_PATH

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="MealWise AI", page_icon="🥗", layout="wide")
st.markdown("""
<style>
.main {background-color:#f7faf7}
[data-testid="stSidebar"] {background-color:#eaf4eb}
h1,h2,h3 {color:#164734}
div[data-testid="stMetric"] {background:#ffffff;padding:16px;border-radius:12px;border:1px solid #d9e7dc}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(path_string):
    return pd.read_csv(path_string)

def load_metrics():
    if METRICS_PATH.exists():
        try: return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

st.sidebar.title("🥗 MealWise AI")
st.sidebar.caption("Predict Better. Prepare Smarter. Waste Less.")
page = st.sidebar.radio("Navigate", ["Home Dashboard", "Data Explorer", "Demand Prediction", "Model Performance", "Food Waste Insights"])
st.sidebar.divider()
st.sidebar.caption("Demo dataset: synthetic, generated for demonstration. Replace with approved anonymized cafeteria records for real use.")

if not DATA_PATH.exists():
    st.error("Dataset not found. Expected file: data/synthetic/meal_attendance_synthetic.csv")
    st.stop()
df = load_data(str(DATA_PATH))
validation_errors = validate_dataset(df)
if validation_errors:
    st.error("Dataset validation issues: " + " | ".join(validation_errors))
    st.stop()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
metrics = load_metrics()

if page == "Home Dashboard":
    st.title("🥗 MealWise AI")
    st.subheader("Predict Better. Prepare Smarter. Waste Less.")
    st.write("A student-friendly machine-learning demo that estimates meal attendance and helps cafeteria teams plan preparation quantities.")
    st.warning("DATA NOTICE: The included dataset is synthetic demonstration data, not actual observations from a college cafeteria. Model scores are not evidence of real-world performance.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Dataset records", f"{len(df):,}")
    c2.metric("Meal types", df["meal_type"].nunique())
    c3.metric("Date range", f"{df['date'].min().date()} – {df['date'].max().date()}")
    st.markdown("### Average attendance by meal")
    summary = df.groupby("meal_type", as_index=False)["attendance"].mean().sort_values("attendance", ascending=False)
    st.bar_chart(summary.set_index("meal_type"))
    st.markdown("### Project workflow")
    st.write("Business understanding → Data understanding → Data preparation → Modeling → Evaluation → Deployment")
    if metrics:
        st.info(f"Selected model: {metrics.get('selected_model','Not trained yet')}. Train/test metrics refer only to the synthetic demonstration dataset.")
    else:
        st.info("Model metrics are not available yet. Run `python -m src.train` from the project folder.")

elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.write("Inspect records, data quality, and attendance patterns.")
    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    dates = st.date_input("Filter date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    meals = st.multiselect("Meal type", sorted(df["meal_type"].unique()), default=sorted(df["meal_type"].unique()))
    view = df[df["meal_type"].isin(meals)]
    if isinstance(dates, (tuple, list)) and len(dates) == 2:
        view = view[(view["date"].dt.date >= dates[0]) & (view["date"].dt.date <= dates[1])]
    st.caption(f"{len(view)} records in current view")
    st.dataframe(view, use_container_width=True)
    left,right = st.columns(2)
    with left:
        st.markdown("**Missing values by column**")
        st.dataframe(df.isna().sum().rename("missing_count"))
    with right:
        st.markdown("**Data quality checks**")
        st.write({"duplicate_rows": int(df.duplicated().sum()), "negative_attendance": int((df["attendance"] < 0).sum()), "invalid_dates": int(df["date"].isna().sum())})
    st.markdown("### Attendance by weekday")
    chart = view.assign(weekday=view["date"].dt.day_name()).groupby("weekday")["attendance"].mean()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    st.bar_chart(chart.reindex(order).dropna())
    st.markdown("### Attendance distribution by meal")
    st.bar_chart(view.groupby("meal_type")["attendance"].mean())

elif page == "Demand Prediction":
    st.title("🔮 Demand Prediction")
    st.write("Choose the planned date, meal, and menu. Prediction requires the saved model artifact.")
    with st.form("prediction_form"):
        pred_date = st.date_input("Date", value=date.today())
        meal = st.selectbox("Meal type", ["breakfast","lunch","dinner"])
        menu = st.selectbox("Menu category", sorted(df["menu_category"].dropna().unique()))
        holiday = st.selectbox("Is it a holiday?", ["No","Yes"])
        margin = st.slider("Safety margin (%)", 0, 30, 5)
        planned = st.number_input("Already planned meals (for comparison)", min_value=0, value=200, step=10)
        submitted = st.form_submit_button("Predict demand")
    if submitted:
        input_df = pd.DataFrame([{"date": str(pred_date), "meal_type": meal, "menu_category": menu, "is_holiday": 1 if holiday=="Yes" else 0}])
        try:
            predicted = predict_attendance(input_df)[0]
            rec = make_recommendation(predicted, margin, int(planned))
            a,b,c = st.columns(3)
            a.metric("Predicted attendance", f"{rec['predicted_attendance']} students")
            b.metric("Recommended preparation", f"{rec['recommended_prepare']} meals")
            c.metric("Planned vs predicted", f"{rec['planned_minus_predicted']:+d}")
            if rec["estimated_surplus"]:
                st.info(f"Estimated plan surplus: {rec['estimated_surplus']} meals.")
            elif rec["estimated_shortage"]:
                st.warning(f"Estimated plan shortage: {rec['estimated_shortage']} meals.")
            else:
                st.success("Planned quantity matches the predicted attendance.")
            st.caption(f"Recommendation formula: round predicted attendance, then add {margin}% safety margin and round up. {rec['note']}")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not generate prediction: {exc}")

elif page == "Model Performance":
    st.title("📈 Model Performance")
    if not metrics:
        st.warning("No trained metrics found. Run `python -m src.train` and refresh this page.")
    else:
        st.warning(metrics.get("warning", "Results depend on the dataset used."))
        st.write(f"**Selected model:** {metrics.get('selected_model','Unknown')}")
        st.write(f"Training records: {metrics.get('train_records','—')} | Held-out test records: {metrics.get('test_records','—')}")
        table = pd.DataFrame(metrics.get("metrics", []))
        if not table.empty:
            st.dataframe(table, use_container_width=True)
            st.caption("MAE and RMSE are attendance-count errors; lower is better. R² measures explained variation and can be negative. Compare all models on the same chronological holdout.")
        for filename, caption in [("actual_vs_predicted.png","Actual vs predicted"),("residual_plot.png","Residual/error plot")]:
            path = ROOT / "models" / filename
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
        st.markdown("**Limitations:** Synthetic data has simplified assumptions. Real performance depends on sufficient historical records, menu changes, attendance policies, holidays, and other local factors.")

elif page == "Food Waste Insights":
    st.title("♻️ Food Waste Insights")
    st.write("This section estimates planning gaps. A surplus estimate is not a measurement of food thrown away.")
    daily = df.groupby(["date","meal_type"], as_index=False)["attendance"].mean()
    daily["example_planned"] = (daily["attendance"] * 1.08).round().astype(int)
    daily["estimated_surplus_vs_attendance"] = (daily["example_planned"] - daily["attendance"].round()).clip(lower=0)
    x,y,z = st.columns(3)
    x.metric("Historical rows", len(daily))
    y.metric("Average attendance", f"{daily['attendance'].mean():.1f}")
    z.metric("Example planned surplus", f"{daily['estimated_surplus_vs_attendance'].mean():.1f} per row")
    st.caption("Illustration only: planned quantity is assumed to be 8% above recorded synthetic attendance. This is not measured waste and is not a model-based historical estimate.")
    st.dataframe(daily.tail(30), use_container_width=True)
    st.markdown("### Practical planning suggestions")
    st.write("- Record planned quantity, actual attendance, and leftovers separately.")
    st.write("- Review demand by meal, weekday, holidays, and menu.")
    st.write("- Re-train with approved anonymized real records and compare with a simple baseline.")
    st.write("- Adjust safety margin according to cafeteria risk tolerance; do not assume one margin suits every meal.")
