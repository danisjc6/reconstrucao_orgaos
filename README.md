Reconstrução 3D de órgãos por radiografias multiangulares
Projeto experimental para reconstrução volumétrica de órgãos ex vivo a partir de radiografias digitais adquiridas durante uma rotação de 360°. O processamento é realizado em Python com o ASTRA Toolbox, utilizando geometria de feixe cônico e reconstrução iterativa acelerada por GPU.

O projeto também avalia a consistência geométrica das projeções por meio de um marcador radiopaco e testa correções individuais com a geometria cone_vec.
Estado atual: estudo metodológico em desenvolvimento. Os volumes obtidos ainda não foram validados contra tomografia computadorizada, medidas anatômicas ou outro padrão de referência. Os resultados não têm finalidade diagnóstica.

Objetivos
- Organizar conjuntos de radiografias DICOM adquiridas em vários ângulos.
- Reconstruir volumes tridimensionais com geometria cone-beam.
- Avaliar possíveis deslocamentos durante a rotação do órgão.
- Representar correções específicas por projeção com cone_vec.
- Comparar de maneira padronizada volumes sem e com correção geométrica.
- Documentar parâmetros e resultados para uso acadêmico e reprodutibilidade.
Conjuntos experimentais
Coração ovino
- 29 projeções em ângulos conhecidos e não uniformes.
- Distância fonte–centro de rotação: 715 mm.
- Distância centro de rotação–detector: 190 mm.
- Distância fonte–detector: 905 mm.
- Magnificação geométrica nominal: aproximadamente 1,266.
- Cassete Carestream CR de 35 × 43 cm.
- Aquisição informada: 60 kVp, 100 mA, 5 mAs e 0,05 s.
- Um arquivo duplicado, 14(1).dcm, foi desconsiderado.
Na inspeção inicial desse conjunto, os DICOMs apresentaram matriz de 3404 × 2704 pixels e espaçamento de 0,086215 mm/pixel. Esses valores não devem ser presumidos para outros experimentos: o programa lê o espaçamento do próprio DICOM.
Coração bovino
- 50 projeções uniformemente distribuídas em 360°.
- Incremento nominal: 7,2°.
- Ângulos: 0°, 7,2°, ..., 352,8°.
- Distância fonte–centro de rotação: 655 mm.
- Distância centro de rotação–detector: 190 mm.
- Distância fonte–detector: 845 mm.
- Magnificação geométrica nominal: aproximadamente 1,290.
- Rotação ao redor do eixo longitudinal do órgão.
- Moeda utilizada como marcador radiopaco.
A configuração selecionada para os testes atuais usa sinal angular negativo e não aplica rotação adicional à matriz DICOM (rotacao_imagem_90 = 0).
Estrutura do projeto
reconstrucao_orgaos/
├── README.md
├── .gitignore
├── reconstruir.py
├── reconstruir_cone_vec.py
├── avaliar_alinhamento.py
├── visualizar_alinhamento.py
├── comparar_volumes.py
├── visualizar_volume.py
├── verificar_dicom.py
├── converter_nifti.py
├── dados/
│   ├── coracao_bovino_01/
│   │   ├── config.json
│   │   ├── marcacoes_moeda.json
│   │   └── dicom/                  # não versionado
│   └── coracao_ovino_01/
│       ├── config.json
│       └── dicom/                  # não versionado
└── resultados/                     # não versionado
Função dos scripts
Arquivo	Finalidade
verificar_dicom.py	Inspeciona dimensões, orientação e metadados dos DICOMs.
reconstruir.py	Executa a reconstrução-controle com geometria cone.
avaliar_alinhamento.py	Registra o centro da moeda, ajusta sua trajetória e calcula resíduos.
visualizar_alinhamento.py	Sobrepõe posições observadas, previstas e vetores de diferença às radiografias.
reconstruir_cone_vec.py	Executa a reconstrução experimental com correções individuais na geometria.
comparar_volumes.py	Compara volumes usando os mesmos cortes e a mesma janela de intensidade.
visualizar_volume.py	Permite inspecionar o arquivo volumétrico .npy.
converter_nifti.py	Converte o volume para NIfTI, quando necessário.


Dependências principais
- Python 3.11
- ASTRA Toolbox com suporte CUDA
- NumPy
- SciPy
- pydicom
- Matplotlib
- NiBabel, apenas para conversão NIfTI
O ambiente Python/Conda não deve ser copiado para dentro do repositório. O instalador do Miniconda também não faz parte do projeto.
Configuração utilizada no teste bovino
Os parâmetros atuais são definidos em dados/coracao_bovino_01/config.json:
{
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
  }
}
Matriz reduzida
O fator de redução diminui o número de linhas e colunas de cada radiografia antes da reconstrução. Com fator 4, cada dimensão é dividida aproximadamente por quatro e a imagem passa a conter cerca de 1/16 do número original de pixels.

O campo físico representado é preservado. Por isso, o espaçamento efetivo entre pixels é multiplicado pelo mesmo fator. A redução torna os testes mais rápidos e diminui o consumo de memória e GPU, mas reduz a resolução espacial disponível.
Pré-processamento e limitações radiométricas
As projeções são limitadas globalmente pelos percentis configurados e normalizadas para uma faixa comum. Esse processamento é qualitativo. Não estavam disponíveis imagens de campo aberto (flat-field) e campo escuro (dark-field), portanto não foi realizada uma conversão fisicamente calibrada das intensidades em integrais de atenuação.

Essa limitação deve ser considerada ao interpretar valores do volume, contraste interno e artefatos.
Reconstrução-controle
A reconstrução-controle utiliza a geometria circular simples cone, com:
- distâncias físicas informadas no config.json;
- todos os ângulos da aquisição;
- SIRT3D_CUDA;
- restrição mínima igual a zero;
- 15 iterações nos testes atuais;
- volume de 256 × 256 × 256 voxels;
- extensão física nominal de 220 mm em cada direção.

Instalação:
## Instalação

Este projeto utiliza Conda para gerenciar o ambiente. Instale previamente
Miniconda ou Anaconda.

## Instalação do ambiente

Este projeto utiliza Python 3.11, ASTRA Toolbox com suporte CUDA e um
ambiente Conda.

O Miniconda deve ser instalado separadamente e não está incluído neste
repositório.

Crie e ative o ambiente:

```bash
conda create -n coracao-astra python=3.11
conda activate coracao-astra


Execução:
python reconstruir.py \
  dados/coracao_bovino_01 \
  --run-name teste_02_sirt15_sinal_negativo
Entre os testes realizados, teste_02_sirt15_sinal_negativo foi adotado como controle para a comparação com cone_vec, pois mantém sinal negativo, 15 iterações e a orientação selecionada. O teste com sinal positivo foi usado apenas para verificar a sensibilidade ao sentido angular. Os parâmetros efetivos de cada execução devem sempre ser confirmados em parametros_utilizados.json.
Avaliação do alinhamento
Marcador radiopaco
O centro da moeda foi localizado nas projeções em que o marcador estava visível. Cada observação foi classificada como:
- completa: moeda inteira e centro considerado confiável;
- parcial: moeda incompleta, com menor confiança;
- ausente: marcador não observado no campo.
As marcações foram armazenadas em marcacoes_moeda.json, evitando a repetição do procedimento a cada ajuste.
Ajustes harmônicos
Como a aquisição completa uma volta, a trajetória da moeda foi descrita por funções periódicas de seno e cosseno. O ajuste é realizado separadamente para as coordenadas horizontal e vertical.

O primeiro harmônico utiliza:
q(θ) = c + a₁ cos(θ) + b₁ sen(θ)
O modelo com primeiro e segundo harmônicos acrescenta:
q(θ) = c + a₁ cos(θ) + b₁ sen(θ)
             + a₂ cos(2θ) + b₂ sen(2θ)
O segundo modelo é mais flexível e acompanhou melhor a trajetória observada. Entretanto, a redução dos resíduos não comprova, isoladamente, que ele represente perfeitamente a geometria física.
Resultado do alinhamento
- A marcação de 15.dcm, em 100,8°, foi retirada do ajuste porque a moeda estava parcial e o ponto era incompatível com a trajetória geral.
- A radiografia 15.dcm permaneceu normalmente entre as 50 projeções da reconstrução.
- Após a retirada, o limite robusto exibido no relatório ficou próximo de 5,1 mm.
- Nenhuma observação restante foi sinalizada como outlier pelo critério utilizado.
- Os maiores resíduos remanescentes ocorreram aproximadamente em 50,4°, 244,8° e 309,6°.
O resíduo é definido por:
resíduo_x = x_observado - x_previsto
resíduo_y = y_observado - y_previsto
Esses resíduos são relativos a um modelo empírico. Eles podem conter deslocamento real, incerteza da localização do marcador e inadequação do próprio modelo.
Correção experimental com cone_vec
A geometria cone_vec representa separadamente, em cada projeção:
- posição da fonte;
- centro do detector;
- direção e tamanho físico das colunas do detector;
- direção e tamanho físico das linhas do detector.
A correção é aplicada ao centro do detector com sinal contrário ao resíduo:
correção_coluna = -resíduo_x
correção_linha  = -resíduo_y
Como os resíduos foram medidos na matriz original e a reconstrução utiliza fator de redução 4, os deslocamentos em pixels são divididos por quatro antes de serem aplicados. O deslocamento físico permanece equivalente.
Critério conservador adotado
No conjunto bovino:
- 25 projeções com moeda completa receberam correção;
- 4 projeções com moeda parcial não receberam correção;
- 21 projeções sem moeda não receberam correção.
Não foi inventada ou interpolada uma correção no intervalo sem marcador. Os DICOMs e seus pixels não são modificados; apenas a geometria fornecida ao ASTRA é alterada.

Execução:
python reconstruir_cone_vec.py \
  dados/coracao_bovino_01 \
  --corrections-csv resultados/coracao_bovino_01/alinhamento/medidas_segundo_harmonico_sem_15.csv \
  --run-name teste_cone_vec_correcao_moeda
Arquivos principais produzidos:
teste_cone_vec_correcao_moeda/
├── volume.npy
├── cortes_centrais.png
├── tabela_definitiva_correcoes.csv
├── medidas_alinhamento_utilizadas.csv
├── geometria_cone_vec.npy
├── metadados_correcao.json
└── parametros_utilizados.json
Comparação dos volumes
A comparação principal utiliza:
- controle: teste_02_sirt15_sinal_negativo/volume.npy;
- corrigido: teste_cone_vec_correcao_moeda/volume.npy.
Execução:
python comparar_volumes.py \
  resultados/coracao_bovino_01/teste_02_sirt15_sinal_negativo/volume.npy \
  resultados/coracao_bovino_01/teste_cone_vec_correcao_moeda/volume.npy \
  --output-dir resultados/coracao_bovino_01/comparacao_controle_cone_vec
O programa utiliza os mesmos índices de corte e uma janela de intensidade comum. Ele gera uma figura com o controle, o volume corrigido e a diferença absoluta.
Métricas obtidas
Métrica	Valor
Forma dos volumes	256 × 256 × 256
Janela comum	0 a 0,0082945
Correlação global	0,8671
MAE global	0,0003349
RMSE global	0,0007472
RMSE normalizado pela janela	9,01%
MAE na região central	0,0003278
MAE na região periférica	0,0003369
Energia de gradiente central — controle	2,0638 × 10⁻⁷
Energia de gradiente central — corrigido	2,1033 × 10⁻⁷


A média do volume aumentou aproximadamente 0,32%, o desvio-padrão aumentou cerca de 1,37% e a energia de gradiente central aumentou aproximadamente 1,92% após a correção.
Interpretação parcial
A correção produziu uma alteração mensurável, mantendo semelhança estrutural global entre os volumes. Foi observada melhora visual discreta da definição do órgão, principalmente no plano coronal, acompanhada por pequeno aumento da energia de gradiente central.

Isso ainda não comprova maior fidelidade anatômica. A energia de gradiente também pode aumentar devido a ruído e estrias. O mapa de diferenças mostrou alterações tanto na região do coração quanto na plataforma e nos artefatos periféricos.
Artefatos e limitações atuais
- Número limitado de projeções em relação a uma tomografia convencional.
- Persistência de estrias radiais, principalmente no plano axial.
- Bandas verticais nos planos coronal e sagital.
- Influência da moeda, da plataforma e de outros materiais radiopacos.
- Ausência do marcador em parte dos ângulos.
- Uso de apenas um marcador, insuficiente para estimar independentemente todas as translações, rotações e variações de escala.
- Ajuste harmônico empírico, sem calibração geométrica completa do sistema.
- Ausência de correção flat-field/dark-field.
- Ausência, até o momento, de referência anatômica externa para validar o volume.
Próximas avaliações
1. Comparar vários cortes ao longo dos três eixos, e não apenas os cortes centrais.
2. Avaliar a estabilidade dos resultados em diferentes números de iterações.
3. Testar 30 iterações com a geometria corrigida, mantendo os demais parâmetros.
4. Investigar estratégias para reduzir a influência da plataforma e do marcador metálico.
5. Avaliar outros métodos iterativos ou regularizados para dados esparsos.
6. Validar dimensões e estruturas contra uma referência independente, quando disponível.
7. Em novas aquisições, utilizar dois ou mais marcadores pequenos, preferencialmente esféricos, em posições conhecidas.
Dados não incluídos no GitHub
Os seguintes itens devem permanecer fora do repositório:
- DICOMs originais;
- resultados e volumes reconstruídos;
- arquivos .npy, .npz, .nii e .nii.gz;
- ambientes Python/Conda;
- instalador do Miniconda;
- caches e configurações locais do editor.
Além do tamanho dos arquivos, DICOMs podem conter metadados sensíveis em seus cabeçalhos. O arquivo .gitignore deve ser verificado antes de cada envio com git status.
Uso científico responsável
As métricas deste projeto descrevem diferenças entre reconstruções. Sem um padrão de referência, elas não medem diretamente a exatidão anatômica. Qualquer conclusão deve distinguir:
- melhora visual percebida;
- alteração matemática do volume;
- redução de artefatos;
- fidelidade anatômica validada.
Até o momento, os resultados sustentam apenas as duas primeiras categorias e uma possível melhora parcial da definição. A validação anatômica permanece como etapa futura.
