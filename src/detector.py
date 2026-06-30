from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def load_model(model_path, conf = 0.3):
    model = AutoDetectionModel.from_pretrained(
        model_type = "ultralytics",
        model_path = model_path,
        confidence_threshold = conf,
        device = "mps"
    )
    return model

def detect(model, img_path, slice_size=256, overlap=0.2): 
    # sahiden gelen model , görüntü yolu, slice boyutu, slicelar ne kadar overlap edecek             
    
    """   SAHI'nin ana fonksiyonu. Görüntüyü 256x256'lık parçalara böler, her
    parçada YOLO çalıştırır, tüm tespitleri birleştirir. NMS ile çakışan   
    bbox'ları temizler. yani drone slice çizgisi ile ikiye bölünürse iki tane bounding boxta olmuş 
    olur o yüzden de yanlış tespite sebep olur. NMS bu çoğaltılmış tespitleri siler. NMS = Non maximum suppression """
    
    result = get_sliced_prediction(                                    
        img_path,                                         
        model,                                                         
        slice_height=slice_size,                          
        slice_width=slice_size,                                        
        overlap_height_ratio=overlap,                     
        overlap_width_ratio=overlap,                                   
    )
                                                                         
    detections = []                                       
    for obj in result.object_prediction_list: # sahinin oluşturduğu listeyi gezer
        bbox = obj.bbox                                                  
        cx = (bbox.minx + bbox.maxx) / 2
        cy = (bbox.miny + bbox.maxy) / 2  # merkez koordinat oluşturur. bounding boxu kullanarak                               
        conf = obj.score.value  # güven skoru alır                             
        detections.append((cx, cy, conf))                              
                                                            
    return detections 
