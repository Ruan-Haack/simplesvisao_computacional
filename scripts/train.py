from ultralytics import YOLO
import os
import torch

def treinar():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data.yaml")
    print(f"Buscando arquivo de configuração em: {data_path}")
   
    # tamanho do modelo: 
    # 'n' (nano) é o mais rápido. 's' (small) é um pouco mais preciso
    
    model = YOLO('yolov8n.pt') 
    #Transfer Learning, onde a IA usa conhecimentos prévios para aprender muito mais rápido.
    model.train(
    data=data_path,
    epochs=20,       # vezes que ele vai ler o dataset 
    imgsz=640,       # resolução das imagens
    batch=5,         # quantas imagens processa por vez 
    fraction=0.9,    # porcentagem do dataset
    name='modelo_pets_v2'
)

if __name__ == '__main__':

    treinar()