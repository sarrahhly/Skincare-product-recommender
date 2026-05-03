import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ast

# ── Page Config ──
st.set_page_config(
    page_title="Skincare Product Recommender",
    page_icon="🧴",
    layout="centered"
)

# ── Load Model and Data ──
@st.cache_resource
def load_model():
    with open('skincare_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('skincare_cleaned.csv')

try:
    bundle   = load_model()
    model    = bundle['model']
    scaler   = bundle['scaler']
    le_target = bundle['le_target']
    feature_cols = bundle['feature_cols']
    all_concerns = bundle['all_concerns']
    df_skin  = load_data()

    # Restore concerns_list column
    def extract_concerns(highlights_str):
        concerns = []
        try:
            items = ast.literal_eval(str(highlights_str))
            for item in items:
                if 'Good for:' in item:
                    concern = item.replace('Good for:', '').strip()
                    concerns.append(concern)
        except:
            pass
        return concerns

    df_skin['concerns_list'] = df_skin['highlights'].apply(extract_concerns)

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# ── Header ──
st.title("Skincare Product Recommender")
st.markdown("Find the best skincare products for your skin concern and budget using Machine Learning.")
st.markdown("---")

# ── Sidebar ──
st.sidebar.header("Your Preferences")

concern = st.sidebar.selectbox(
    "Select Your Skin Concern",
    sorted([c for c in all_concerns if c not in ['Color Care', 'Damage', 'Flaky/Dry Scalp',
                                                   'Frizz', 'Hair Thinning', 'Oily Scalp', 'Volume']])
)

max_price = st.sidebar.slider(
    "Maximum Budget (USD)",
    min_value=5,
    max_value=300,
    value=50,
    step=5
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=3,
    max_value=10,
    value=5
)

st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("1. Filter products by your concern")
st.sidebar.markdown("2. Filter by your budget")
st.sidebar.markdown("3. XGBoost predicts product tier")
st.sidebar.markdown("4. Top products ranked by smart score")

# ── Main Button ──
if st.button("Get Recommendations", type="primary", use_container_width=True):

    # Filter by concern
    def has_concern(concerns_list, target):
        try:
            return any(target.lower() in c.lower() for c in concerns_list)
        except:
            return False

    filtered = df_skin[
        df_skin['concerns_list'].apply(lambda x: has_concern(x, concern)) &
        (df_skin['price_usd'] <= max_price)
    ].copy()

    if filtered.empty:
        st.warning(f"No products found for '{concern}' under ${max_price}. Try increasing your budget.")
    else:
        # Predict
        filtered_feat   = filtered[feature_cols].copy()
        filtered_scaled = scaler.transform(filtered_feat)
        predictions     = model.predict(filtered_scaled)
        filtered['predicted_tier'] = le_target.inverse_transform(predictions)

        # Get High first, fallback to Medium
        top = filtered[filtered['predicted_tier'] == 'High']
        if top.empty:
            top = filtered[filtered['predicted_tier'] == 'Medium']
            st.info("No High tier products found in this range. Showing Medium tier products.")

        top = top.sort_values('smart_score', ascending=False).head(top_n)

        # ── Results ──
        st.markdown(f"### Top {len(top)} Recommendations")
        st.markdown(f"**Concern:** {concern} | **Budget:** ${max_price} | **Matches found:** {len(filtered)}")
        st.markdown("---")

        for i, (_, row) in enumerate(top.iterrows(), 1):
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**{i}. {row['product_name']}**")
                    st.markdown(f"Brand: {row['brand_name']}")
                    st.markdown(f"Category: {row.get('secondary_category', 'N/A')}")

                with col2:
                    st.metric("Price", f"${row['price_usd']:.2f}")
                    st.metric("Rating", f"{row['rating']:.1f} / 5")

                col3, col4, col5 = st.columns(3)
                with col3:
                    st.markdown(f"Loves: **{int(row['loves_count']):,}**")
                with col4:
                    st.markdown(f"Reviews: **{int(row['reviews']):,}**")
                with col5:
                    tier = row['predicted_tier']
                    color = "green" if tier == "High" else "orange" if tier == "Medium" else "red"
                    st.markdown(f"Tier: **:{color}[{tier}]**")

                st.markdown("---")

        # ── Summary Stats ──
        st.markdown("### Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Products Found", len(filtered))
        with col2:
            st.metric("Avg Price", f"${top['price_usd'].mean():.2f}")
        with col3:
            st.metric("Avg Rating", f"{top['rating'].mean():.2f}")

# ── About Section ──
st.markdown("---")
with st.expander("About This App"):
    st.markdown("""
    This app uses a Machine Learning model trained on the **Sephora Products and Skincare Reviews** dataset from Kaggle.

    **ML Pipeline:**
    - Data cleaning and feature engineering on 8,216 skincare products
    - Smart score created from rating (50%) + loves (30%) + reviews (20%)
    - XGBoost classifier trained to predict product quality tier
    - Products filtered by skin concern and budget, then ranked by smart score

    **Models Compared:**
    - Logistic Regression (baseline)
    - Decision Tree
    - Random Forest
    - XGBoost (best model - selected for this app)

    **Available Skin Concerns:**
    Acne/Blemishes, Anti-Aging, Dark Circles, Dark spots, Dryness,
    Dullness/Uneven Texture, Loss of firmness, Pores, Redness
    """)
