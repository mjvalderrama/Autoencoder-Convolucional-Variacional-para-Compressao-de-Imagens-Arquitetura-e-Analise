Compressão de Imagens com Autoencoder Convolucional Variacional (CVAE)
Implementação de um codec de imagem neural baseado em um CVAE profundo para compressão com perdas, desenvolvido como parte de um projeto para a disciplina de Processamento de Imagens Digitais do Programa de Pós-Graduação em Ciência da Computação (PPGCC) da UNESP.

Este repositório contém os códigos completos para o treinamento, compressão (encoding) e descompressão (decoding) de imagens utilizando um modelo CVAE treinado no dataset MNIST. O trabalho completo, incluindo a análise teórica e dos resultados, está documentado no artigo acadêmico associado.

Arquitetura do Modelo
O modelo implementado é um Convolutional Variational Autoencoder (CVAE). A arquitetura combina a eficiência de camadas convolucionais para processamento de dados espaciais com a estrutura probabilística de um VAE para aprender uma representação latente regularizada e compacta. A Tabela abaixo detalha a arquitetura:

Componente

Camada

Especificação (Filtros/Neurónios, Kernel, Strides, Ativação)

Input

Input

Shape: (28, 28, 1)

Encoder

Conv2D

32 filtros, kernel (3,3), strides (2,2), padding 'same', ReLU



Conv2D

64 filtros, kernel (3,3), strides (2,2), padding 'same', ReLU



Flatten

-



Dense

128 neurónios, ReLU



Dense (Saídas)

Duas saídas: z_mean (16 neurónios) e z_log_var (16 neurónios)

Latent Space

Lambda (Sampling)

Amostra de N(µ, σ²) com 16 dimensões (Reparameterization Trick)

Decoder

Input

Shape: (16,)



Dense

7 x 7 x 64 = 3136 neurónios, ReLU



Reshape

Target Shape: (7, 7, 64)



Conv2DTranspose

64 filtros, kernel (3,3), strides (2,2), padding 'same', ReLU



Conv2DTranspose

32 filtros, kernel (3,3), strides (2,2), padding 'same', ReLU



Conv2DTranspose (Saída)

1 filtro, kernel (3,3), padding 'same', Sigmoid

Resultados Visuais
Abaixo, um exemplo da compressão e reconstrução de um dígito do dataset MNIST.


Exemplo da imagem de um dígito '3' após ser comprimida para um vetor de 16 dimensões e reconstruída pelo CVAE.

Estrutura do Repositório
Compress-o_Neural_de_Imagens/
│
├── CVAE_train.py           # Script para treinar o modelo CVAE
├── CVAE_encoder.py         # Script para comprimir (codificar) uma imagem
├── CVAE_decoder.py         # Script para descomprimir (decodificar) uma imagem
│
├── CVAE_encoder_train.h5   # (Gerado após o treino) Modelo do encoder salvo
├── CVAE_decoder_train.h5   # (Gerado após o treino) Modelo do decoder salvo
│
├── digit_3.png             # Imagem de exemplo para teste
├── digit_3_compressed.npy  # (Gerado pelo encoder) Vetor comprimido
├── digit_3_reconstructed.png # (Gerado pelo decoder) Imagem reconstruída
│
└── README.md               # Este ficheiro

Instalação e Configuração
Pré-requisitos
Python 3.8+

TensorFlow / Keras

NumPy

Pillow (PIL)

Passos
Clone o repositório:

git clone https://github.com/mjvalderrama/Compress-o_Neural_de_Imagens.git
cd Compress-o_Neural_de_Imagens

Crie um ambiente virtual (recomendado):

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

Instale as dependências:

pip install tensorflow numpy pillow matplotlib

Como Usar
O pipeline completo consiste em três etapas:

1. Treinamento do Modelo
Execute o script de treinamento para gerar os ficheiros dos modelos CVAE_encoder_train.h5 e CVAE_decoder_train.h5. O dataset MNIST será descarregado automaticamente pelo Keras.

python CVAE_train.py

Este processo pode demorar, dependendo do seu hardware (uma GPU é altamente recomendada).

2. Compressão de uma Imagem (Encoding)
Após o treinamento, use o encoder para comprimir uma imagem. Certifique-se de que a imagem de entrada (ex: digit_3.png) esteja na pasta.

python CVAE_encoder.py

Isto irá gerar o ficheiro digit_3_compressed.npy, que é a sua imagem comprimida.

3. Descompressão da Imagem (Decoding)
Use o decoder para reconstruir a imagem a partir do ficheiro comprimido.

python CVAE_decoder.py

Isto irá gerar o ficheiro digit_3_reconstructed.png, que é a imagem final reconstruída.

Como Citar Este Trabalho
Se utilizar este código ou os conceitos deste trabalho na sua pesquisa, por favor, cite o nosso artigo:

@inproceedings{Rossi2025CVAE,
  title     = {Autoencoder Convolucional Variacional para Compressão de Imagens: Arquitetura e Análise},
  author    = {Rossi, Pedro Martins and Roncolletta, Leonardo and Valderrama, Marcio Jos{\'e}},
  booktitle = {Trabalho de Disciplina, PPGCC-UNESP},
  year      = {2025},
  organization = {Universidade Estadual Paulista (UNESP)}
}

Autores
Pedro Martins Rossi

Leonardo Roncolletta

Marcio José Valderrama

Agradecimentos
Agradecemos ao Prof. Dr. Leandro Alves Neves pela orientação e apoio durante a disciplina de Processamento de Imagens Digitais (PPGCC-UNESP), que motivou este trabalho.

Licença
Este projeto é distribuído sob a licença MIT. Veja o ficheiro LICENSE para mais detalhes.
