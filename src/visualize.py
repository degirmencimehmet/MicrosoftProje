import cv2

COLORS = {  # BGR formatında olması önemli çünkü YOLO bgr formatında çalışıyor
    "KILITLI": (0, 255, 0),  # Yeşil
    "TAHMIN": (0, 165, 255), # Turuncu
    "ARANIYOR": (255, 0, 0), # Mavi
    "KAYIP": (0, 0, 255)     # Kırmızı
}

def draw_tracks(frame, tracks):
    for track in tracks:
        x, y, vx, vy = track.kalman.state
        x, y = int(x), int(y)

        color = COLORS.get(track.state.name, (255, 255, 255))

        cv2.rectangle(frame, (x-10, y-10), (x+10, y+10), color, 2)  # y-10 görüntüde farklı işler o yüzden sol üst değil sol aalt köşe olur

        label = f"ID:{track.id} {track.state.name}"
        cv2.putText(frame, label, (x-10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 )

    return frame