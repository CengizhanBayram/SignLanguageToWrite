import cv2

# Video akışını aç
cap = cv2.VideoCapture(0)  # 0, 1, 2 gibi farklı değerlerle farklı kameraları seçebilirsiniz.

# Başlangıç penceresinin seçimi
ret, frame = cap.read()
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
tracker = cv2.TrackerKCF_create()
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()

    # Nesneyi takip et
    success, bbox = tracker.update(frame)

    # Başarı durumunu kontrol et
    if success:
        # Nesneyi çerçeve içine al
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure detected!", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Görüntüyü göster
    cv2.imshow("Object Tracking", frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizleme
cap.release()
cv2.destroyAllWindows()
