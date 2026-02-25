from ultralytics import YOLO
import os
import torch

def treinar():
    # tamanho do modelo: 
    # 'n' (nano) é o mais rápido. 's' (small) é um pouco mais preciso
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data.yaml")
    print(f"Buscando arquivo de configuração em: {data_path}")
    
    model = YOLO('yolov8n.pt') 
    model.train(
        data=data_path,   
        epochs=10,          # Quantas vezes a IA vai ler o dataset
        imgsz=640,          # Resolução das imagens
        batch=4,            # Quantas imagens processa por vez 
        fraction=0.3,
        name='meu_modelo_pets_rapido'
    )

if __name__ == '__main__':

    treinar()