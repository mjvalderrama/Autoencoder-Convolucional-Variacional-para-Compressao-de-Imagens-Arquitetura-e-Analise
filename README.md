# Autoencoder Convolucional Variacional para Compressao de Imagens Arquitetura e Análise


Implementação de um codec de imagem neural baseado em um CVAE profundo para compressão com perdas, desenvolvido como parte de um projeto para a disciplina de **Processamento de Imagens Digitais** do Programa de Pós-Graduação em Ciência da Computação (PPGCC) da UNESP.

Este repositório contém os códigos completos para o **treinamento**, **compressão (encoding)** e **descompressão (decoding)** de imagens utilizando um modelo CVAE treinado no dataset MNIST. O trabalho completo, incluindo a análise teórica e dos resultados, está documentado no artigo acadêmico associado.



## 🧠 Arquitetura do Modelo

O modelo implementado é um **Convolutional Variational Autoencoder (CVAE)**. A arquitetura combina a eficiência de camadas convolucionais para processamento de dados espaciais com a estrutura probabilística de um VAE para aprender uma representação latente regularizada e compacta.

## ARQUITETURA E PARÂMETROS DO CVAE

A tabela a seguir detalha a arquitetura e os hiperparâmetros do modelo implementado.

| Componente | Função | Configuração |
| :--- | :--- | :--- |
| **Entrada** | Dataset MNIST | Banco de dados modificado do *National Institute of Standards and Technology* |
| **Encoder** | Camada Convolucional | 32 Filtros, Kernel 3x3, Ativação ReLU, Stride 2, Padding 'same' |
| | Camada Convolucional | 64 Filtros, Kernel 3x3, Ativação ReLU, Stride 2, Padding 'same' |
| | Camada Totalmente Conectada | 128 Neurônios, Ativação ReLU |
| | Camada Totalmente Conectada | Saída para a Média (μ) |
| | Camada Totalmente Conectada | Saída para o Desvio Padrão (σ) |
| **Espaço Latente** | Dimensões | 16 |
| **Decoder** | Camada Totalmente Conectada | 7x7x64, Ativação ReLU |
| | Camada Convolucional Transposta | 64 Filtros, Kernel 3x3, Ativação ReLU, Stride 2, Padding 'same' |
| | Camada Convolucional Transposta | 32 Filtros, Kernel 3x3, Ativação ReLU, Stride 2, Padding 'same' |
| | Camada Convolucional Transposta | 1 Filtro, Kernel 3x3, Ativação Sigmoid, Stride 1, Padding 'same' |
| **Cálculo da Função de Perda** | Função de Perda | Divergência de Kullback-Leibler + Entropia Cruzada Binária |
| **Aprendizado Iterativo** | Otimizador | Adam |
| **Hiperparâmetros** | Tamanho do Lote (Batch Size) | 128 |
| | Épocas (Epochs) | 50 |

## 🖼️ Resultados Visuais

Abaixo, um exemplo da compressão e reconstrução de um dígito do dataset MNIST:

- **Esquerda:** Imagem original  
- **Direita:** Imagem reconstruída pelo CVAE após ser comprimida para um vetor de apenas 16 dimensões.

---

## 📁 Estrutura do Repositório

```text
Compress-o_Neural_de_Imagens/
│
├── CVAE_train.py              # Script para treinar o modelo CVAE
├── CVAE_encoder.py            # Script para comprimir (codificar) uma imagem
├── CVAE_decoder.py            # Script para descomprimir (decodificar) uma imagem
│
├── CVAE_encoder_train.h5      # (Gerado após o treino) Modelo do encoder salvo
├── CVAE_decoder_train.h5      # (Gerado após o treino) Modelo do decoder salvo
│
├── digit_3.png                # Imagem de exemplo para teste
├── digit_3_compressed.npy     # (Gerado pelo encoder) Vetor comprimido
├── digit_3_reconstructed.png  # (Gerado pelo decoder) Imagem reconstruída
│
└── README.md                  # Este ficheiro


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
