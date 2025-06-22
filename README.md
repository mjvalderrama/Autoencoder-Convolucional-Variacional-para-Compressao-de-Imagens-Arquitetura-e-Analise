# Compressão de Imagens com Autoencoder Convolucional Variacional (CVAE)

Implementação de um codec de imagem neural baseado em um CVAE profundo para compressão com perdas, desenvolvido como parte de um projeto para a disciplina de **Processamento de Imagens Digitais** do Programa de Pós-Graduação em Ciência da Computação (PPGCC) da UNESP.

Este repositório contém os códigos completos para o **treinamento**, **compressão (encoding)** e **descompressão (decoding)** de imagens utilizando um modelo CVAE treinado no dataset MNIST. O trabalho completo, incluindo a análise teórica e dos resultados, está documentado no artigo acadêmico associado.

---

## 🧠 Arquitetura do Modelo

O modelo implementado é um **Convolutional Variational Autoencoder (CVAE)**. A arquitetura combina a eficiência de camadas convolucionais para processamento de dados espaciais com a estrutura probabilística de um VAE para aprender uma representação latente regularizada e compacta.

| Componente     | Camada           | Especificação |
|----------------|------------------|---------------|
| **Input**      | InputShape       | (28, 28, 1) |
| **Encoder**    | Conv2D           | 32 filtros, kernel (3x3), strides (2x2), padding 'same', ReLU |
|                | Conv2D           | 64 filtros, kernel (3x3), strides (2x2), padding 'same', ReLU |
|                | Flatten + Dense  | 128 neurônios, ReLU |
|                | Dense (Saídas)   | `z_mean` (16 neurônios) e `z_log_var` (16 neurônios) |
| **Latent Space** | Lambda (Sampling) | Amostra de N(µ, σ²) com 16 dimensões (Reparameterization Trick) |
| **Decoder**    | InputShape       | (16,) |
|                | Dense            | 3136 neurônios (7x7x64), ReLU |
|                | Reshape          | (7, 7, 64) |
|                | Conv2DTranspose  | 64 filtros, kernel (3x3), strides (2x2), padding 'same', ReLU |
|                | Conv2DTranspose  | 32 filtros, kernel (3x3), strides (2x2), padding 'same', ReLU |
|                | Saída            | 1 filtro, kernel (3x3), padding 'same', Sigmoid |

---

## 🖼️ Resultados Visuais

Abaixo, um exemplo da compressão e reconstrução de um dígito do dataset MNIST:

- **Esquerda:** Imagem original  
- **Direita:** Imagem reconstruída pelo CVAE após ser comprimida para um vetor de apenas 16 dimensões.

---

## 📁 Estrutura do Repositório

