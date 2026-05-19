import streamlit as st
import numpy as np
from PIL import Image
import joblib
import keras

# load models
pca = joblib.load('chest_Xray/pca.pkl')
best_lr = joblib.load('chest_Xray/lr.pkl')
best_r = joblib.load('chest_Xray/r.pkl')
best_svm = joblib.load('chest_Xray/best_svm.pkl')
cnn_model = keras.models.load_model('chest_Xray/best_cnn.keras')

st.title('🫁 Zoidberg 2.0 — Pneumonia Detection')
st.write('Upload a chest X-ray image and all 4 models will predict the diagnosis.')

uploaded_file = st.file_uploader('Choose an X-ray image', type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L').resize((64, 64))
    img_array = np.array(img) / 255.0
    
    st.image(img, caption='Uploaded X-ray', width=200)
    
    # classical ML
    img_flat = img_array.flatten().reshape(1, -1)
    img_pca = pca.transform(img_flat)
    
    lr_pred = best_lr.predict(img_pca)[0]
    lr_prob = best_lr.predict_proba(img_pca)[0][1]
    
    rf_pred = best_r.predict(img_pca)[0]
    rf_prob = best_r.predict_proba(img_pca)[0][1]
    
    svm_pred = best_svm.predict(img_pca)[0]
    svm_prob = best_svm.predict_proba(img_pca)[0][1]
    
    # CNN
    img_cnn = img_array.reshape(1, 64, 64, 1)
    cnn_prob = cnn_model.predict(img_cnn, verbose=0)[0][0]
    cnn_pred = 1 if cnn_prob > 0.5 else 0
    
    def label(pred, prob):
        if pred == 1:
            return f'🔴 PNEUMONIA ({prob:.0%})'
        else:
            return f'🟢 NORMAL ({(1-prob):.0%})'
    
    st.subheader('Results:')
    st.write(f'**Logistic Regression:** {label(lr_pred, lr_prob)}')
    st.write(f'**Random Forest:** {label(rf_pred, rf_prob)}')
    st.write(f'**SVM:** {label(svm_pred, svm_prob)}')
    st.write(f'**CNN:** {label(cnn_pred, cnn_prob)}')