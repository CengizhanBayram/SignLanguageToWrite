# 1. Gerekli Kütüphanelerin Yüklenmesi
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time # Eğitim süresini ölçmek için

# 2. Parametrelerin ve Ayarların Tanımlanması
# -----------------------------------------------------------------------------
# Veri Seti Ayarları
DATA_PATH = os.path.join('MP_Veri_Pytorch') # Verilerin kaydedileceği/okunacağı klasör
actions = np.array(['merhaba', 'tesekkurler', 'lutfen']) # Tanınacak işaret sınıfları
no_sequences = 30  # Her bir işaret için toplanacak video (örnek) sayısı
sequence_length = 30  # Her bir videodaki kare (frame) sayısı
# MediaPipe Holistic'ten gelen toplam eklem noktası sayısı (örneğin 33*4 + 468*3 + 21*3 + 21*3)
# Bu değeri kendi veri toplama yönteminize göre ayarlayın.
num_landmarks = 1662 

# Model Hiperparametreleri
INPUT_SIZE = num_landmarks     # Modelin girdi boyutu (her karedeki toplam veri noktası)
HIDDEN_SIZE = 128              # LSTM katmanındaki nöron (bellek hücresi) sayısı
NUM_LAYERS = 2                 # Üst üste yığılacak LSTM katmanı sayısı (daha derin öğrenme için)
NUM_CLASSES = len(actions)     # Çıktı sayısı (sınıf sayısı)
LEARNING_RATE = 0.001          # Optimizasyon algoritmasının öğrenme oranı
BATCH_SIZE = 16                # Eğitim sırasında her adımda modele verilecek örnek sayısı
NUM_EPOCHS = 200               # Eğitim veri setinin üzerinden kaç kez geçileceği

# Cihaz Ayarı (GPU varsa kullanılır, yoksa CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 3. Örnek Veri Seti Oluşturma ve Yükleme
# -----------------------------------------------------------------------------
# NOT: Bu bölüm, projenin çalışmasını sağlamak için rastgele veri üretir.
# Gerçek bir uygulamada, bu bölümü kendi veri toplama ve işleme kodunuzla değiştirmelisiniz.
# MediaPipe ile topladığınız numpy array'lerini bu klasöre kaydetmeniz yeterlidir.
sequences_path = os.path.join(DATA_PATH, 'sequences.npy')
labels_path = os.path.join(DATA_PATH, 'labels.npy')

if not os.path.exists(sequences_path):
    print("Mevcut veri seti bulunamadı. Örnek veri seti oluşturuluyor...")
    os.makedirs(DATA_PATH, exist_ok=True)
    sequences, labels = [], []
    for action_idx, action in enumerate(actions):
        for _ in range(no_sequences):
            window = []
            for _ in range(sequence_length):
                # Rastgele 3D koordinatlar oluştur
                res = np.random.rand(num_landmarks).astype(np.float32)
                window.append(res)
            sequences.append(window)
            labels.append(action_idx)
            
    np.save(sequences_path, np.array(sequences))
    np.save(labels_path, np.array(labels))
    print(f"Veri seti oluşturuldu ve '{DATA_PATH}' klasörüne kaydedildi.")

# Kaydedilmiş veriyi yükle
X = np.load(sequences_path)
y = np.load(labels_path)

# Veriyi eğitim ve test setlerine ayır (stratify=y, sınıfların oranını korur)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print(f"\nVeri seti yüklendi. Toplam Örnek: {len(X)}")
print(f"Eğitim Seti Boyutu: {X_train.shape}")
print(f"Test Seti Boyutu: {X_test.shape}")


# 4. PyTorch için Özel Veri Seti Sınıfı ve DataLoader
# -----------------------------------------------------------------------------
class SignLanguageDataset(Dataset):
    """İşaret dili verilerini PyTorch tensörlerine dönüştüren özel Dataset sınıfı."""
    def __init__(self, features, labels):
        # Verileri ve etiketleri alıp torch tensörlerine dönüştürür.
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # Veri setindeki toplam örnek sayısını döndürür.
        return len(self.features)

    def __getitem__(self, idx):
        # Belirli bir indeksteki (idx) veri ve etiketi döndürür.
        return self.features[idx], self.labels[idx]

# DataLoader'ları oluştur. Bu yapılar veriyi modele verimli bir şekilde besler.
train_dataset = SignLanguageDataset(X_train, y_train)
test_dataset = SignLanguageDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 5. RNN-LSTM Model Mimarisi
# -----------------------------------------------------------------------------
class SignLSTM(nn.Module):
    """İşaret dili tanıma için LSTM tabanlı sinir ağı modeli."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SignLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Katmanı:
        # - input_size: Her zaman adımındaki özellik sayısı (1662)
        # - hidden_size: LSTM hücrelerinin sayısı (hafıza kapasitesi)
        # - num_layers: Üst üste LSTM katmanı sayısı
        # - batch_first=True: Girdi tensörünün boyutunu (batch, seq, feature) yapar.
        # - dropout: Katmanlar arasına dropout ekleyerek aşırı öğrenmeyi (overfitting) önler.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Sınıflandırma için Tam Bağlantılı (Fully Connected) Katmanlar
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2) # LSTM çıktısını daha küçük bir boyuta indirger.
        self.relu = nn.ReLU() # Aktivasyon fonksiyonu
        self.fc2 = nn.Linear(hidden_size // 2, num_classes) # Son sınıflandırma katmanı
        self.dropout = nn.Dropout(0.3) # Sınıflandırma katmanları arasında dropout

    def forward(self, x):
        # LSTM için başlangıç gizli (hidden) ve hücre (cell) durumlarını ayarla
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Veriyi LSTM katmanından geçir
        out, _ = self.lstm(x, (h0, c0))
        
        # Sadece son zaman adımının çıktısını al (tüm sekansın bir özetidir)
        out = out[:, -1, :]
        
        # Tam bağlantılı katmanlardan geçir
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) # Son çıktı (logits)
        
        return out


# 6. Modelin, Kayıp Fonksiyonunun ve Optimizatörün Hazırlanması
# -----------------------------------------------------------------------------
model = SignLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)

# Kayıp Fonksiyonu: Çok sınıflı sınıflandırma için CrossEntropyLoss idealdir.
# İçerisinde Softmax fonksiyonunu barındırdığı için modelin sonuna ayrıca eklemeye gerek yoktur.
criterion = nn.CrossEntropyLoss()

# Optimizatör: Ağırlıkları güncellemek için Adam algoritmasını kullanıyoruz.
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel Mimarisi:\n{model}")
print(f"\nModel '{device}' cihazı üzerinde çalışacak.")


# 7. Eğitim Fonksiyonu
# -----------------------------------------------------------------------------
def train_model():
    print("\n--- Model Eğitimi Başlatılıyor ---")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train() # Modeli eğitim moduna al (Dropout gibi katmanları aktif eder)
        epoch_loss = 0
        for i, (sequences, labels) in enumerate(train_loader):
            # Verileri seçilen cihaza (GPU/CPU) gönder
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # İleri Yayılım (Forward pass)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Geri Yayılım ve Optimizasyon (Backward and optimize)
            optimizer.zero_grad() # Her adımdan önce gradyanları sıfırla
            loss.backward()       # Kayba göre gradyanları hesapla
            optimizer.step()      # Hesaplanan gradyanlara göre ağırlıkları güncelle
            
            epoch_loss += loss.item()
        
        # Her 10 epoch'ta bir ilerleme durumu yazdır
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Ortalama Kayıp (Loss): {avg_loss:.4f}')
            
    end_time = time.time()
    print(f"--- Eğitim Tamamlandı. Süre: {(end_time - start_time)/60:.2f} dakika ---")


# 8. Değerlendirme Fonksiyonu
# -----------------------------------------------------------------------------
def check_accuracy():
    print("\n--- Model Test Verisi Üzerinde Değerlendiriliyor ---")
    model.eval() # Modeli değerlendirme moduna al (Dropout gibi katmanları deaktif eder)
    with torch.no_grad(): # Gradyan hesaplamayı durdurarak belleği ve hızı optimize et
        n_correct = 0
        n_samples = 0
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            # En yüksek değere sahip sınıfın indeksini al (tahmin edilen sınıf)
            _, predicted = torch.max(outputs.data, 1)
            
            n_samples += labels.size(0) # Toplam etiket sayısını artır
            n_correct += (predicted == labels).sum().item() # Doğru tahmin sayısını artır
            
        acc = 100.0 * n_correct / n_samples
        print(f'Modelin Test Verisindeki Doğruluğu: {acc:.2f} %')


# 9. Ana Çalıştırma Bloğu ve Modeli Kaydetme
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Modeli eğit
    train_model()

    # 2. Modelin doğruluğunu kontrol et
    check_accuracy()

    # 3. Eğitilmiş modeli ileride kullanmak üzere kaydet
    MODEL_PATH = "sign_language_lstm_model.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nEğitilmiş model '{MODEL_PATH}' dosyasına başarıyla kaydedildi.")

    # Tahmin için modeli yükleme ve kullanma örneği:
    # --------------------------------------------------
    # print("\nKaydedilmiş model ile tahmin örneği:")
    # loaded_model = SignLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    # loaded_model.load_state_dict(torch.load(MODEL_PATH))
    # loaded_model.eval()

    # # Test setinden bir örnek alıp tahmin yapalım
    # sample_sequence, sample_label = test_dataset[0]
    # sample_sequence = sample_sequence.unsqueeze(0).to(device) # Batch boyutu ekle: (30, 1662) -> (1, 30, 1662)

    # with torch.no_grad():
    #     prediction_output = loaded_model(sample_sequence)
    #     _, predicted_idx = torch.max(prediction_output.data, 1)
    
    # predicted_action = actions[predicted_idx.item()]
    # true_action = actions[sample_label.item()]

    # print(f"Gerçek İşaret: {true_action}")
    # print(f"Tahmin Edilen İşaret: {predicted_action}")
