import cv2 # OpenCV (Open Source Computer Vision Library)
from ultralytics import YOLO

class VisionApp:
    def __init__(self, model_path):
        """
        Construtor da Instância: Carrega o modelo YOLOv8.
        """
        self.model = YOLO(model_path)
        self.last_detection = "Nenhum"

    def run(self):
        """
        Inicia a captura e desenha o BBox no animal detectado.
        """
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # inferência supervisionada
            results = self.model(frame, conf=0.6, verbose=False)
            
            found = False
            for r in results:
                for box in r.boxes:
                    # coordenadas do BBox (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # identificação da classe
                    cls_id = int(box.cls[0])
                    self.last_detection = "Gato" if cls_id == 0 else "Cachorro"
                    found = True
                    
                    # Bouding Box
                    
                    # cv2.rectangle(imagem, ponto1, ponto2, cor, espessura)
                    cor_bbox = (0, 200, 0) 
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), cor_bbox, 2)
                    
                    # colocar o nome do animal em cima da caixa
                    cv2.putText(frame, self.last_detection, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, cor_bbox, 2)
                    break 
            
            if not found:
                self.last_detection = "Nenhum"

            # HUD de Status
            cv2.putText(frame, f"Status: {self.last_detection}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Teste de Identificacao com BBox", frame)
            
            # COMANDO PARA MATAR O PROGRAMA: 's' ou Esc
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == 27:
                print("Encerrando programa...")
                break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Criando a instância e rodando
    caminho_modelo = "runs/detect/modelo_pets_v2/weights/best.pt"
    instancia_teste = VisionApp(caminho_modelo)
    instancia_teste.run()