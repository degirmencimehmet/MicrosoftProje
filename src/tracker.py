from enum import Enum
from src.kalman import kalmanFilter
from src.association import associate

class State(Enum):  # enum kullanma sebebimiz string kısımları int değerlere atamak için
    ARANIYOR = 0
    KILITLI = 1
    TAHMIN = 2
    KAYIP = 3

class Track:
    def __init__(self, x, y, track_id):
        self.id = track_id
        self.kalman = kalmanFilter(x,y)
        self.state = State.KILITLI
        self.missing_count = 0  # kaç frame boyunca yolo droneu göremedi o amaçla kullanılır
                                # stateler arası geçişte lazım mesela count 3 oldu kayıp durumuna geçer

    def predict(self):
        return self.kalman.predict()
    
    def update(self, x,y): # state değiştiğinde ARANIYOR -> KILITLI ye geçişte countu ve statei değiştirmemiz lazım
        self.kalman.update(x,y)
        self.missing_count = 0 
        self.state= State.KILITLI

class Tracker:
    def __init__(self, max_missing = 3 ):
        self.tracks = []
        self.next_id = 0
        self.max_missing = max_missing

    def update(self, detections):
        predictions = []
        for t in self.tracks:
            predictions.append(t.predict())    # tüm trackleri tahmin et

        matches , unmatched_detections = associate(predictions , detections) # eşleştirme yapıldı

        for t_index , d_index in matches :
            self.tracks[t_index].update(*detections[d_index])
        
        matched_track_ids = [t_idx for t_idx, _ in matches]
        for i, track in enumerate(self.tracks):
            if i not in matched_track_ids:
                track.missing_count += 1
                track.state = State.TAHMIN
                if track.missing_count > self.max_missing:
                    track.state = State.KAYIP

        # 5. Eşleşmeyen tespitler → yeni track oluştur
        for d_idx in unmatched_detections:
            x, y = detections[d_idx]
            self.tracks.append(Track(x, y, self.next_id))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.state !=State.KAYIP]

        return self.tracks

        
