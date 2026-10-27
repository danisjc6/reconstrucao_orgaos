# Reconstrução 3D de órgãos com ASTRA Toolbox

Projeto em Python para reconstrução volumétrica experimental de órgãos a partir de radiografias DICOM multiangulares, utilizando geometria cone-beam e o ASTRA Toolbox com aceleração CUDA.

## Requisitos

- Ubuntu ou outra distribuição Linux compatível;
- GPU NVIDIA com driver e suporte CUDA;
- Miniconda ou Anaconda;
- Python 3.11;
- ASTRA Toolbox;
- NumPy, SciPy, Matplotlib, pydicom e NiBabel.

O instalador do Miniconda, os DICOMs e os volumes reconstruídos não fazem parte do repositório.

## Instalação

Instale previamente o [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Em seguida, crie o ambiente:

```bash
conda create -n coracao-astra python=3.11
conda activate coracao-astra
```

Instale o ASTRA Toolbox:

```bash
conda install -c astra-toolbox -c nvidia astra-toolbox
```

Instale as demais dependências:

```bash
pip install numpy scipy matplotlib pydicom nibabel
```

Verifique a instalação e o acesso à GPU:

```bash
python -c "import astra; print('ASTRA:', astra.__version__); print('CUDA:', astra.use_cuda())"
```

O resultado esperado para CUDA é:

```text
CUDA: True
```

## Organização dos dados

Crie uma pasta para cada experimento:

```text
dados/
└── coracao_bovino_01/
    ├── config.json
    └── dicom/
        ├── 1.dcm
        ├── 2.dcm
        └── ...
```

Os arquivos DICOM devem ser numerados conforme a ordem dos ângulos. A pasta `dicom/` está excluída do Git pelo `.gitignore`.

## Configuração do experimento

Exemplo de `dados/coracao_bovino_01/config.json`:

```json
{
  "nome": "coracao_bovino_01",
  "numero_projecoes": 50,
  "angulos": {
    "tipo": "uniforme",
    "inicial_graus": 0.0,
    "total_graus": 360.0,
    "sinal": -1.0
  },
  "geometria": {
    "distancia_fonte_orgao_mm": 655.0,
    "distancia_orgao_detector_mm": 190.0
  },
  "orientacao": {
    "rotacao_imagem_90": 0
  },
  "processamento": {
    "fator_reducao": 4,
    "percentil_inferior": 0.5,
    "percentil_superior": 99.5
  },
  "volume": {
    "tamanho_voxels": 256,
    "extensao_mm": 220.0
  },
  "reconstrucao": {
    "algoritmo": "SIRT3D_CUDA",
    "iteracoes": 15,
    "restricao_minima": 0.0
  },
  "alinhamento": {
    "avaliar": true,
    "aplicar_correcao": false,
    "referencia": "moeda"
  }
}
```

### Parâmetros principais

| Campo | Descrição |
| --- | --- |
| `numero_projecoes` | Quantidade de radiografias do experimento. |
| `angulos.tipo` | `uniforme` ou `lista`. |
| `angulos.sinal` | Sentido angular usado na geometria. |
| `distancia_fonte_orgao_mm` | Distância entre a fonte e o centro de rotação. |
| `distancia_orgao_detector_mm` | Distância entre o centro de rotação e o detector. |
| `rotacao_imagem_90` | Número de rotações de 90° aplicadas aos DICOMs. |
| `fator_reducao` | Redução de linhas e colunas para diminuir o custo computacional. |
| `tamanho_voxels` | Dimensão cúbica do volume reconstruído. |
| `extensao_mm` | Extensão física do volume em cada direção. |
| `iteracoes` | Número de iterações do SIRT3D. |

Para ângulos não uniformes, use:

```json
"angulos": {
  "tipo": "lista",
  "valores_graus": [0, 10, 35, 50, 60],
  "sinal": -1.0
}
```

## Execução básica

Ative o ambiente:

```bash
conda activate coracao-astra
```

Reconstrução com geometria `cone`:

```bash
python reconstruir.py \
  dados/coracao_bovino_01 \
  --run-name teste_sirt15
```

Reconstrução corrigida com `cone_vec`:

```bash
python reconstruir_cone_vec.py \
  dados/coracao_bovino_01 \
  --corrections-csv caminho/para/medidas_alinhamento.csv \
  --run-name teste_cone_vec
```

Os resultados são gravados em:

```text
resultados/<nome_do_experimento>/<nome_da_execucao>/
```

As pastas `resultados/` e `dados/*/dicom/` devem permanecer fora do GitHub.

