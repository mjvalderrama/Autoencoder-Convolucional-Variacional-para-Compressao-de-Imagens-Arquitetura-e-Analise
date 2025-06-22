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


Compress-o\_Neural\_de\_Imagens/
│
├── CVAE\_train.py               # Script para treinar o modelo CVAE
├── CVAE\_encoder.py             # Script para comprimir (codificar) uma imagem
├── CVAE\_decoder.py             # Script para descomprimir (decodificar) uma imagem
│
├── CVAE\_encoder\_train.h5      # (Gerado após o treino) Modelo do encoder salvo
├── CVAE\_decoder\_train.h5      # (Gerado após o treino) Modelo do decoder salvo
│
├── digit\_3.png                 # Imagem de exemplo para teste
├── digit\_3\_compressed.npy     # (Gerado pelo encoder) Vetor comprimido
├── digit\_3\_reconstructed.png  # (Gerado pelo decoder) Imagem reconstruída
│
└── README.md                    # Este ficheiro

````

---

## ⚙️ Instalação e Configuração

### ✅ Pré-requisitos

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Pillow (PIL)

### 📥 Passos

```bash
# Clone o repositório
git clone https://github.com/mjvalderrama/Compress-o_Neural_de_Imagens.git
cd Compress-o_Neural_de_Imagens

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install tensorflow numpy pillow matplotlib
````

---

## ▶️ Como Usar

O pipeline completo consiste em **três etapas**:

### 1. Treinamento do Modelo

Execute o script de treinamento para gerar os arquivos `CVAE_encoder_train.h5` e `CVAE_decoder_train.h5`. O dataset MNIST será baixado automaticamente.

```bash
python CVAE_train.py
```

> Este processo pode demorar, dependendo do seu hardware (uma GPU é altamente recomendada).

---

### 2. Compressão de uma Imagem (Encoding)

Após o treinamento, use o encoder para comprimir uma imagem:

```bash
python CVAE_encoder.py
```

> Este comando utiliza o arquivo `digit_3.png` como entrada e gera `digit_3_compressed.npy`.

---

### 3. Descompressão da Imagem (Decoding)

Reconstrua a imagem a partir do vetor comprimido:

```bash
python CVAE_decoder.py
```

> Este comando gera `digit_3_reconstructed.png` como saída.

---

## 📚 Como Citar Este Trabalho

Se utilizar este código ou os conceitos deste trabalho na sua pesquisa, por favor, cite:

```bibtex
@inproceedings{Rossi2025CVAE,
  title     = {Autoencoder Convolucional Variacional para Compressão de Imagens: Arquitetura e Análise},
  author    = {Rossi, Pedro Martins and Roncolletta, Leonardo and Valderrama, Marcio Jos{\'e}},
  booktitle = {Trabalho de Disciplina, PPGCC-UNESP},
  year      = {2025},
  organization = {Universidade Estadual Paulista (UNESP)}
}
```

---

## 👥 Autores

* Pedro Martins Rossi
* Leonardo Roncolletta
* Marcio José Valderrama

---

## 🙏 Agradecimentos

Agradecemos ao **Prof. Dr. Leandro Alves Neves** pela orientação e apoio durante a disciplina de Processamento de Imagens Digitais (PPGCC-UNESP), que motivou este trabalho.

---

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

```

Se quiser, posso gerar o arquivo `.md` pronto para você fazer o upload direto no GitHub. Deseja isso?
```
