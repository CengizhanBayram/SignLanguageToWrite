import tensorflow as tf
import mediapipe as mp
import numpy as np

# Modelin yüklenmesi
model = tf.keras.models.load_model("isareet_dili_modeli.h5")

# MediaPipe ellerin tespiti için kullanılır
hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# Kelime haznesi ve dil modeli
# ... (KenLM modelini ve kelime haznesini yükleme kodunu buraya ekleyin)

# Dönüştürme fonksiyonu
def işaret_dili_ceviri(frame):
    # Ellerin tespiti
    results = hands.process(frame)

    if results.multi_hand_landmarks:  # Eğer eller tespit edilmişse
        hand_landmarks = results.multi_hand_landmarks[0]  # İlk elin iskeleti

        # El özelliklerini çıkarma
        # Her bir elin koordinatlarını alarak bir özellik vektörü oluşturabiliriz
        hand_features = []
        for landmark in hand_landmarks.landmark:
            hand_features.append(landmark.x)
            hand_features.append(landmark.y)
            hand_features.append(landmark.z if landmark.z else 0)  # Eğer el tespit edilmezse z ekseni bilgisi olmayabilir

        # Özellik vektörünü numpy dizisine dönüştürme
        features = np.array([hand_features])

        # Tahminleme
        predictions = model.predict(features)

        # Tahminleri yazıya çevirme
        predicted_text = tahminleri_yaziya_cevir(predictions)

        # Görselleştirme
        # El iskeletlerini çizme
        image_height, image_width, _ = frame.shape
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * image_width), int(lm.y * image_height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Tahminleri gösterme
        cv2.putText(frame, predicted_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return predicted_text

    else:  # Eller tespit edilmezse
        return "Eller tespit edilemedi"

def tahminleri_yaziya_cevir(predictions):
    if predictions is not None:
        # Tahminlerin hangi sınıflara karşılık geldiğini belirleme
        # ... (KenLM modelini kullanarak sınıfları belirleme kodunu buraya ekleyin)

        # En yüksek olasılıklı sınıfın kelimeye dönüştürülmesi
        # ... (Kelime haznesini kullanarak kelimeyi bulma kodunu buraya ekleyin)

        return "Tahmin"
    else:
        return "Tahmin edilemedi"

def main():
    # Kameradan görüntü yakalama
    cap = cv2.VideoCapture(0)

    while True:
        # Kameradan görüntü yakalama
        ret, frame = cap.read()

        # İşaret dilini yazıya çevirme
        predicted_text = işaret_dili_ceviri(frame)

        # Görüntüleri gösterme
        cv2.imshow('İşaret Dili Çevirisi', frame)

        # 'q' tuşuna basılmasını bekleyin
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
