from flask import Flask, request, render_template
import pickle
import numpy as np

# Flask uygulamasını başlat
app = Flask(__name__)

# Modeli ve scaler'ı yükleyin
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Sınıf etiketlerini tanımlayın
class_labels = {
    0: "Anemia",
    1: "Diabet",
    2: "Healthy",
    3: "Thalassemia"
}

# Ana sayfa için rota
@app.route('/')
def index():
    return render_template('index.html')

# Tahmin için rota
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Formdan gelen veriyi al
        data = request.form.to_dict()
        input_data = np.array([float(value) for value in data.values()])
        
        # Veriyi ölçeklendirme
        input_data_scaled = scaler.transform([input_data])
        
        # Tahmini yapın
        prediction = model.predict(input_data_scaled)[0]
        
        # Sınıf etiketini belirleyin
        prediction_label = class_labels.get(prediction, "Unknown")
        
        return render_template('index.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
