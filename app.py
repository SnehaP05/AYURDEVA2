# ============================================================
# AyurHealth AI — Streamlit App (10 Diseases, Checkbox Inputs)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os
from datetime import datetime
from fpdf import FPDF
from dosha_engine import (identify_dosha, get_dosha_for_disease,
                           get_recommendation, DOSHA_INFO)

st.set_page_config(page_title="AyurHealth AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.main-title{font-size:2.4rem;font-weight:700;color:#2D6A4F;text-align:center}
.sub-title{font-size:1.1rem;color:#52796F;text-align:center;margin-bottom:1.5rem}
.disease-box{background:#FFF3CD;border-radius:10px;padding:1.2rem;text-align:center}
.dosha-card{background:#F0F7F4;border-radius:12px;padding:1rem;border-left:5px solid #2D6A4F}
.herb-item{background:#E8F5E9;border-radius:8px;padding:0.5rem 0.8rem;margin:4px 0}
.wellness{background:#2D6A4F;color:white;border-radius:16px;padding:2rem;
          text-align:center;font-size:1.4rem;font-weight:600;margin-top:2rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model      = joblib.load('disease_model.pkl')
        scaler     = joblib.load('scaler.pkl')
        le_gender  = joblib.load('le_gender.pkl')
        le_disease = joblib.load('le_disease.pkl')
        with open('disease_classes.json') as f:
            classes = json.load(f)
        if os.path.exists('feature_cols.json'):
            with open('feature_cols.json') as f:
                feat_cols = json.load(f)
        else:
            feat_cols = ['Age','Gender_enc','BMI','BP','Sugar','Cholesterol',
                         'Thyroid','Smoking','Asthma','Stress',
                         'Fatigue','JointPain','Headache','Nausea','SkinIssue']
        return model, scaler, le_gender, le_disease, classes, feat_cols
    except FileNotFoundError:
        return None, None, None, None, None, None

model, scaler, le_gender, le_disease, disease_classes, FEATURE_COLS = load_model()

st.markdown('<div class="main-title">🌿 AyurHealth AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ancient Ayurvedic Wisdom · XGBoost AI · 10 Diseases</div>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("👤 Patient Details")
    name   = st.text_input("Full Name", placeholder="Enter your name")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        age = st.slider("Age", 10, 90, 35)

    location = st.selectbox("Location", [
        "Mumbai","Delhi","Kolkata","Chennai","Bangalore",
        "Hyderabad","Pune","Ahmedabad","Jaipur","Lucknow","Other"
    ])

    st.divider()
    st.subheader("🩺 Clinical Measurements")
    bmi         = st.slider("BMI",                   10.0, 50.0, 24.0, step=0.1)
    bp          = st.slider("Blood Pressure (mmHg)",  80,   210,  120)
    sugar       = st.slider("Blood Sugar (mg/dL)",     70,   380,  100)
    cholesterol = st.slider("Cholesterol (mg/dL)",    100,   310,  180)
    stress      = st.slider("Stress Level (1-10)",      1,    10,    5)

    st.divider()
    st.subheader("Medical History")
    st.caption("Select all that apply")
    thyroid = st.checkbox("Thyroid condition")
    smoking = st.checkbox("Current smoker")
    asthma  = st.checkbox("Asthma diagnosed")

    st.divider()
    st.subheader("Current Symptoms")
    st.caption("Select all you are experiencing")
    fatigue    = st.checkbox("Fatigue / tiredness")
    joint_pain = st.checkbox("Joint pain / stiffness")
    headache   = st.checkbox("Headache / migraine")
    nausea     = st.checkbox("Nausea / acidity")
    skin_issue = st.checkbox("Skin issues / hair loss")

    st.divider()
    symptoms = st.text_area("Additional symptoms (optional)",
        placeholder="e.g. dry skin, frequent urination...", height=80)

    predict_btn = st.button("Predict and Recommend",
                             use_container_width=True, type="primary")

if predict_btn:
    if model is None:
        st.error("Model files not found! Upload all .pkl and .json files to GitHub.")
        st.stop()

    gender_enc = le_gender.transform([gender])[0]
    features   = np.array([[
        age, gender_enc, bmi, bp, sugar, cholesterol,
        int(thyroid), int(smoking), int(asthma), stress,
        int(fatigue), int(joint_pain), int(headache),
        int(nausea), int(skin_issue)
    ]])
    features_scaled   = scaler.transform(features)
    disease_enc       = model.predict(features_scaled)[0]
    proba             = model.predict_proba(features_scaled)[0]
    predicted_disease = le_disease.inverse_transform([disease_enc])[0]
    confidence        = proba[disease_enc] * 100

    top5_idx = np.argsort(proba)[::-1][:5]
    top5     = [(le_disease.inverse_transform([i])[0], proba[i]*100) for i in top5_idx]

    computed_dosha, dosha_scores = identify_dosha(age, bmi, bp, stress, smoking, symptoms)
    final_dosha    = get_dosha_for_disease(predicted_disease, computed_dosha)
    recommendation = get_recommendation(predicted_disease)
    dosha_info     = DOSHA_INFO[final_dosha]

    st.header("Prediction Results")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="disease-box">
            <h2 style='color:#C0392B'>⚠️ {predicted_disease}</h2>
            <p style='font-size:1.1rem'>Confidence: <b>{confidence:.1f}%</b></p>
            <p style='color:#888;font-size:0.8rem'>AI prediction — not a medical diagnosis</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Your Selected Symptoms:**")
        symptom_map = {
            "Fatigue": fatigue, "Joint Pain": joint_pain, "Headache": headache,
            "Nausea": nausea, "Skin Issue": skin_issue,
            "Thyroid": thyroid, "Smoker": smoking, "Asthma": asthma
        }
        active = [k for k, v in symptom_map.items() if v]
        if active:
            for s in active:
                st.markdown(f"✅ {s}")
        else:
            st.caption("No symptoms selected")

    with c2:
        st.markdown(f"""
        <div class="dosha-card" style="background:{dosha_info['color']}">
            <h3>Dosha: {final_dosha}</h3>
            <p><b>Elements:</b> {dosha_info['elements']}</p>
            <p><b>Qualities:</b> {dosha_info['qualities']}</p>
            <p><b>Imbalance signs:</b> {dosha_info['imbalance_signs']}</p>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.subheader("Top 5 Predictions")
        for disease, prob in top5:
            st.progress(int(prob), text=f"{disease} — {prob:.1f}%")

    st.divider()
    cd1, cd2 = st.columns(2)
    with cd1:
        st.subheader("Dosha Balance")
        fig_radar = go.Figure(go.Scatterpolar(
            r=[dosha_scores['Vata'], dosha_scores['Pitta'], dosha_scores['Kapha']],
            theta=['Vata','Pitta','Kapha'], fill='toself',
            line_color='#2D6A4F', fillcolor='rgba(45,106,79,0.25)'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                showlegend=False, height=300,
                                margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    with cd2:
        st.subheader("Prediction Confidence")
        fig_bar = px.bar(
            x=[p for _,p in top5], y=[d for d,_ in top5], orientation='h',
            color=[p for _,p in top5], color_continuous_scale='Greens',
            labels={'x':'Probability (%)','y':''}
        )
        fig_bar.update_layout(height=300, showlegend=False,
                               coloraxis_showscale=False,
                               margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.header("Personalised Ayurvedic Recommendations")
    tab1, tab2, tab3, tab4 = st.tabs(["Herbs and Remedy","Diet Plan","Yoga and Exercise","Lifestyle"])

    with tab1:
        h1, h2 = st.columns(2)
        with h1:
            st.subheader("Recommended Herbs")
            for herb in recommendation['herbs']:
                st.markdown(f"<div class='herb-item'>🌿 {herb}</div>", unsafe_allow_html=True)
        with h2:
            st.subheader("Organic Remedy")
            st.info(recommendation['remedy'])

    with tab2:
        for tip in recommendation['diet']:
            prefix = "✅" if any(tip.startswith(w) for w in ["Eat","Add","Pair","Increase"]) else \
                     "❌" if any(tip.startswith(w) for w in ["Avoid","Never","Limit"]) else "💡"
            st.markdown(f"{prefix} {tip}")

    with tab3:
        for pose in recommendation['yoga']:
            st.markdown(f"🧘 {pose}")

    with tab4:
        for tip in recommendation['lifestyle']:
            st.markdown(f"🌙 {tip}")

    st.divider()
    st.header(f"Location Analysis — {location}")

    prevalence_data = {
        'Mumbai':    {'Diabetes':22,'Hypertension':28,'Asthma':18,'Thyroid Disorder':15,'Obesity':12,'Others':5},
        'Delhi':     {'Diabetes':20,'Hypertension':30,'Asthma':22,'Thyroid Disorder':12,'Obesity':10,'Others':6},
        'Kolkata':   {'Diabetes':25,'Hypertension':22,'Asthma':15,'Thyroid Disorder':20,'Anemia':12,'Others':6},
        'Chennai':   {'Diabetes':30,'Hypertension':20,'Asthma':12,'Thyroid Disorder':22,'Obesity':10,'Others':6},
        'Bangalore': {'Diabetes':22,'Hypertension':20,'Anxiety Disorder':18,'Thyroid Disorder':20,'GERD':14,'Others':6},
        'Hyderabad': {'Diabetes':28,'Hypertension':22,'Obesity':15,'Thyroid Disorder':18,'GERD':12,'Others':5},
        'Pune':      {'Diabetes':20,'Hypertension':20,'Anxiety Disorder':20,'GERD':18,'Thyroid Disorder':16,'Others':6},
        'Other':     {'Diabetes':22,'Hypertension':22,'Asthma':18,'Thyroid Disorder':18,'Others':20},
    }
    prev = prevalence_data.get(location, prevalence_data['Other'])

    lc1, lc2 = st.columns(2)
    with lc1:
        fig_pie = px.pie(names=list(prev.keys()), values=list(prev.values()),
                         title=f"Disease Prevalence in {location}",
                         color_discrete_sequence=px.colors.sequential.Greens_r)
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    with lc2:
        st.subheader("Regional Stats")
        for disease, pct in prev.items():
            st.metric(label=disease, value=f"{pct}%")
        if predicted_disease in prev:
            st.warning(f"⚠️ {prev[predicted_disease]}% of people in {location} are affected by {predicted_disease}.")

    st.divider()
    st.subheader("Age vs Disease Prevalence")
    age_labels = ['18-25','25-35','35-45','45-55','55-65','65-75']
    age_data = {
        'Diabetes':         [ 2,  6, 15, 28, 36, 42],
        'Hypertension':     [ 1,  3,  9, 22, 36, 48],
        'Asthma':           [12, 10,  8,  6,  5,  4],
        'Thyroid Disorder': [ 5, 14, 20, 18, 12,  8],
        'Anxiety Disorder': [14, 18, 12,  8,  5,  3],
        'Arthritis':        [ 1,  2,  6, 18, 34, 52],
        'Migraine':         [10, 16, 14,  8,  5,  3],
        'Obesity':          [ 6, 12, 18, 20, 16, 10],
    }
    age_df  = pd.DataFrame(age_data, index=age_labels)
    fig_age = px.line(age_df, markers=True,
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      labels={'index':'Age Group','value':'Prevalence (%)','variable':'Disease'})
    fig_age.update_layout(height=380)
    st.plotly_chart(fig_age, use_container_width=True)

    st.divider()
    st.header("Health Report")

    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica","B",20)
        pdf.cell(0,12,"AyurHealth AI — Personal Health Report",ln=True,align='C')
        pdf.set_font("Helvetica","",11)
        pdf.cell(0,8,f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",ln=True,align='C')
        pdf.ln(4)
        def section(t):
            pdf.set_fill_color(45,106,79); pdf.set_text_color(255,255,255)
            pdf.set_font("Helvetica","B",13)
            pdf.cell(0,9,f"  {t}",ln=True,fill=True)
            pdf.set_text_color(0,0,0); pdf.set_font("Helvetica","",11); pdf.ln(2)
        def row(l,v):
            pdf.set_font("Helvetica","B",11); pdf.cell(60,8,l+":",ln=False)
            pdf.set_font("Helvetica","",11); pdf.cell(0,8,str(v),ln=True)
        section("Patient Information")
        row("Name",name or "Not provided"); row("Age",f"{age} years")
        row("Gender",gender); row("Location",location); pdf.ln(3)
        section("Clinical Measurements")
        row("BMI",f"{bmi} kg/m2"); row("BP",f"{bp} mmHg")
        row("Sugar",f"{sugar} mg/dL"); row("Cholesterol",f"{cholesterol} mg/dL")
        row("Stress",f"{stress}/10"); pdf.ln(3)
        section("Medical History and Symptoms")
        row("Thyroid","Yes" if thyroid else "No"); row("Smoker","Yes" if smoking else "No")
        row("Asthma","Yes" if asthma else "No"); row("Fatigue","Yes" if fatigue else "No")
        row("Joint Pain","Yes" if joint_pain else "No"); row("Headache","Yes" if headache else "No")
        row("Nausea","Yes" if nausea else "No"); row("Skin Issues","Yes" if skin_issue else "No"); pdf.ln(3)
        section("Prediction"); row("Disease",predicted_disease); row("Confidence",f"{confidence:.1f}%"); row("Dosha",final_dosha); pdf.ln(3)
        section("Herbs")
        for h in recommendation['herbs']: pdf.multi_cell(0,7,f"  * {h}")
        pdf.ln(2); section("Remedy"); pdf.multi_cell(0,7,recommendation['remedy']); pdf.ln(2)
        section("Diet")
        for t in recommendation['diet']: pdf.multi_cell(0,7,f"  * {t}")
        pdf.ln(2); section("Yoga")
        for p in recommendation['yoga']: pdf.multi_cell(0,7,f"  * {p}")
        pdf.ln(2); section("Lifestyle")
        for t in recommendation['lifestyle']: pdf.multi_cell(0,7,f"  * {t}")
        pdf.ln(4)
        pdf.set_font("Helvetica","BI",14); pdf.set_text_color(45,106,79)
        pdf.cell(0,12,"Stay healthy, be happy, take care!",ln=True,align='C')
        pdf.set_font("Helvetica","I",9); pdf.set_text_color(128,128,128)
        pdf.cell(0,8,"Disclaimer: AI-generated for informational purposes only.",ln=True,align='C')
        return bytes(pdf.output())

    if st.button("Generate PDF Health Report", use_container_width=True):
        with st.spinner("Generating..."):
            pdf_bytes = generate_pdf()
        st.download_button("Download PDF", data=pdf_bytes,
            file_name=f"AyurHealth_{name or 'Report'}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf", use_container_width=True)

    st.markdown("""
    <div class="wellness">
        🌿 Stay healthy, be happy, take care 😊<br>
        <span style='font-size:0.9rem;font-weight:400;opacity:0.85'>
        Ayurveda treats the whole person — body, mind, and spirit.
        </span>
    </div>""", unsafe_allow_html=True)
    st.caption("Disclaimer: For educational purposes only. Always consult a certified medical professional.")

else:
    st.info("👈 Fill in your details in the sidebar and click Predict and Recommend")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Diseases","10 conditions")
    c2.metric("Model","XGBoost")
    c3.metric("Doshas","Vata · Pitta · Kapha")
    c4.metric("Report","PDF download")
    st.markdown("""
    ### 10 Diseases Covered
    | Disease | Key Indicators |
    |---|---|
    | Diabetes | High sugar, high BMI |
    | Hypertension | High BP, older age, stress |
    | Asthma | Asthma flag, smoking, fatigue |
    | Thyroid Disorder | Thyroid flag, fatigue, skin issues |
    | Anxiety Disorder | High stress, headache, nausea |
    | Obesity | Very high BMI, fatigue, joint pain |
    | Anemia | Low BMI, fatigue, female skew |
    | GERD | High BMI, stress, nausea |
    | Migraine | High stress, headache, female skew |
    | Arthritis | Older age, high BMI, joint pain |
    """)
    if model is None:
        st.warning("Model files not found. Upload all .pkl and .json files to GitHub.")
