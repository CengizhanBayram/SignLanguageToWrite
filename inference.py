

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

# --- 1. Model ve Veri Parametrelerinin Tanımlanması ---
# ÖNEMLİ: Bu parametreler, modeli eğitirken kullandığınız parametrelerle
# BİREBİR AYNI olmalıdır.
actions = np.array(['merhaba', 'tesekkurler', 'lutfen'])
sequence_length = 30
num_landmarks = 1662
MODEL_PATH = "sign_language_lstm_model.pth"

# Model Hiperparametreleri
INPUT_SIZE = num_landmarks
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = len(actions)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 2. Model Mimarisi Tanımı ---
# Ağırlıkları yükleyebilmek için eğitimdekiyle aynı mimariyi tanımlıyoruz.
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# --- 3. Yardımcı Fonksiyonlar ---

def load_model(model_path):
    """Verilen yoldan eğitilmiş modeli yükler ve tahmin için hazırlar."""
    print(f"'{model_path}' yolundan model yükleniyor...")
    model = SignLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model başarıyla yüklendi.")
        return model
    except FileNotFoundError:
        print(f"HATA: Model dosyası bulunamadı! Lütfen önce modeli eğitin.")
        return None

def extract_keypoints(results):
    """MediaPipe sonuçlarından eklem noktalarını tek bir numpy dizisine dönüştürür."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# --- 4. Canlı Demo Ana Kodu ---

# Modeli yükle
model = load_model(MODEL_PATH)
if model is None:
    exit()

# MediaPipe araçlarını hazırla
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Kamera akışını başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("HATA: Kamera başlatılamadı.")
    exit()

# Tahmin için kullanılacak değişkenler
sequence = []
current_prediction = ''
prediction_threshold = 0.8 # Tahminin gösterilmesi için gereken minimum olasılık

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Kameradan bir kare oku
        success, frame = cap.read()
        if not success:
            print("Kamera akışı alınamıyor. Çıkılıyor...")
            break

        # Görüntüyü MediaPipe için işle
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Performans için görüntüyü yazılamaz yap
        results = holistic.process(image)
        image.flags.writeable = True # Görüntüyü tekrar yazılabilir yap
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ekrana iskeleti çizdir
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Tahmin mantığı
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:] # Diziyi her zaman 30 karede tut

        if len(sequence) == sequence_length:
            # Sekans tamamlandığında tahmin yap
            input_tensor = torch.tensor(np.expand_dims(sequence, axis=0), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                # Softmax ile olasılıkları hesapla
                probabilities = torch.softmax(outputs, dim=1)[0]
                
            # En yüksek olasılığa sahip sınıfı al
            top_prob, top_idx = probabilities.topk(1)
            
            if top_prob.item() > prediction_threshold:
                current_prediction = actions[top_idx.item()]
            else:
                current_prediction = "..." # Eşik altında ise belirsiz

        # Tahmin sonucunu ekrana yazdır
        # Sol üste bir dikdörtgen çiz
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        # Metni yazdır
        cv2.putText(image, f"TAHMIN: {current_prediction.upper()}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Görüntüyü göster
        cv2.imshow('Canli Isaret Dili Tanima', image)

        # 'q' tuşuna basıldığında döngüden çık
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
