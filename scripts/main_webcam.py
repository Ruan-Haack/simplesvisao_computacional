import cv2 # OpenCV (Open Source Computer Vision Library)
from ultralytics import YOLO

class VisionApp:
    def __init__(self, model_path):
        """
        Construtor da Instância: Carrega o modelo YOLOv8
        """
        self.model = YOLO(model_path)
        self.last_detection = "Nenhum"

    def run(self):
        """
        Inicia a captura e desenha o bbox no animal detectado
        """
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # inferência supervisionada
            results = self.model(frame, conf=0.7, verbose=False)
            
            found = False
            for r in results:
                for box in r.boxes:
                    # coordenadas do BBox (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    confianca = float(box.conf[0]) * 100
                    
                    # identificação da classe
                    cls_id = int(box.cls[0])
                    label = "Gato" if cls_id == 0 else "Cachorro"
                    found = True
                    
                    texto_exibicao = f"{label} {confianca:.1f}%"
                    
                    # Bouding Box
                    
                    # cv2.rectangle(imagem, ponto1, ponto2, cor, espessura)
                    cor_bbox = (0, 200, 0) 
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), cor_bbox, 2)
                    cv2.putText(frame, texto_exibicao, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, cor_bbox, 2)
                   
                    break 
            
            if not found:
                self.last_detection = "Nenhum"

            # HUD de Status
            cv2.putText(frame, f"Status: {self.last_detection}", (20, 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow("Teste de Identificacao com BBox", frame)
            
            # COMANDO PARA MATAR O PROGRAMA: 's' ou Esc
            key = cv2.waitKey(5) & 0xFF
            if key == ord('s') or key == 27:
                print("Encerrando programa...")
                break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    caminho_modelo = "runs/detect/modelo_pets_v4/weights/best.pt"
    instancia_teste = VisionApp(caminho_modelo)
    instancia_teste.run()