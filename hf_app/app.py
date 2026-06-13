import gradio as gr
import numpy as np
from PIL import Image
import joblib
from tensorflow import keras

# load models
pca = joblib.load('pca.pkl')
best_lr = joblib.load('lr.pkl')
best_r = joblib.load('r.pkl')
best_svm = joblib.load('best_svm.pkl')
cnn_model = keras.models.load_model('best_cnn.keras')

def predict(image):
    img = Image.fromarray(image).convert('L').resize((64, 64))
    img_array = np.array(img) / 255.0
    
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
    
    return (
        f"**Logistic Regression:** {label(lr_pred, lr_prob)}\n\n"
        f"**Random Forest:** {label(rf_pred, rf_prob)}\n\n"
        f"**SVM:** {label(svm_pred, svm_prob)}\n\n"
        f"**CNN:** {label(cnn_pred, cnn_prob)}"
    )

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='numpy', label='Upload Chest X-Ray'),
    outputs=gr.Markdown(label='Predictions'),
    title='🫁 Zoidberg 2.0 — Pneumonia Detection',
    description='Upload a chest X-ray image. All 4 models will predict NORMAL or PNEUMONIA with confidence percentage.',
)

demo.launch()