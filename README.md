# 🐾 IA Vision: Detecção e Centralização de Pets com YOLOv8

Este projeto faz parte da minha trilha de estudos em **Visão Computacional** e **Robótica**. O objetivo é desenvolver um sistema capaz de detectar gatos e cachorros em tempo real e calcular métricas de centralização de alvo, simulando a lógica necessária para o pouso autônomo de drones.

## 🚀 Tecnologias e Ferramentas
* **Linguagem:** Python 3.10.12
* **Framework de IA:** Ultralytics YOLOv8 (Modelo Nano)
* **Processamento de Imagem:** OpenCV
* **Ambiente:** Ubuntu Linux (Desenvolvimento focado em performance para CPU)
* **Dataset:** Oxford-IIIT Pet Dataset (Subconjunto customizado)

## 📊 Resultados do Treinamento Supervisionado
O treinamento foi realizado utilizando a técnica de *Transfer Learning* sobre o modelo pré-treinado `yolov8n.pt`. Com foco em eficiência para hardware embarcado, apliquei as seguintes configurações:

| Parâmetro | Valor |
| :--- | :--- |
| **Épocas** | 10 |
| **Tamanho da Imagem (imgsz)** | 640px |
| **Batch Size** | 4 |
| **Precisão Média (mAP50)** | **93.2%** |
| **Latência de Inferência (CPU)** | ~82.8ms |

Os resultados demonstram que, mesmo com um treinamento "express" (usando 20% do dataset), o modelo atingiu uma precisão superior a 90%, sendo capaz de diferenciar raças variadas de gatos e cachorros.

## 🤖 Lógica de Centralização (Robótica)
Diferente de uma detecção comum, este projeto implementa um overlay de engenharia que calcula o erro de centralização (`dx`, `dy`):
- **Alvo Centralizado:** Quando o objeto entra na margem de segurança de 80px, o sistema valida a prontidão para ação.
- **Feedback Visual:** Mudança dinâmica de cores no HUD (Heads-Up Display) para indicar o status da detecção.

## 📂 Estrutura do Repositório
* `scripts/train.py`: Script automatizado para treinamento com caminhos absolutos.
* `scripts/organizarlabel.py`: Utilitário para normalização de labels do Oxford-Pets.
* `data.yaml`: Configuração do mapeamento do dataset.
* `runs/`: Logs de treinamento e métricas (Matriz de Confusão, Gráficos de Perda).

## 🛠️ Como Executar
1. Clone o repositório.
2. Crie e ative o ambiente virtual: `source venv/bin/activate`.
3. Instale as dependências: `pip install -r requirements.txt`.
4. Execute a inferência: `python3 scripts/webcam_test.py`.

---
**Desenvolvido por Ruan Haack** *Graduando em Sistemas de Informação - UNEB | Pesquisador em Robótica e Data Science*