# Alerta Chuva e Alagamento SP · Visão Computacional

**GS 2026.2 · CitizenTech · Applied Computer Vision, "Enxergando o mundo real"**
Professores: Arnaldo Viana e Paulo Rodrigues

Um detector YOLO, treinado pela equipe, que reconhece **rua alagada**, **veículo ilhado** e **pessoa na água** em fotos e em vídeo ao vivo. Cada detecção vira um **nível de risco** (Sem alagamento, Atenção, Alto, Crítico) que alimenta os relatos de alagamento do MVP Alerta Chuva SP.

- **Demonstração online (roda no celular):** `https://zbaldd.github.io/GS-VISAO-COMPUTACIONAL/web/`
- **MVP (mapa de alertas):** `https://zbaldd.github.io/GS-BIGDATA-2026/mvp/`
- **Vídeo da solução funcionando:** `LINK_DO_VIDEO`

## Integrantes

| Nome completo | RM |
|---|---|
| Arthur Baldissera Claumann Marcos | RM550219 |
| Victor Kenzo Toma| RM551649 |
| Ricardo Ramos Vergani | 	RM550166 |
| Pedro Barros | RM97937 |

## Ficha Técnica do MVP

| Item | Conteúdo |
|---|---|
| **Nome do produto** | Alerta Chuva SP |
| **Cidadão e território** | Moradores e trabalhadores que circulam pela Grande São Paulo (10 municípios) em dias de chuva forte |
| **Problema, em uma frase** | Quem sai de casa em dia de temporal não sabe quais ruas estão alagadas e acaba ilhado, com prejuízo e risco de vida |
| **Proposta de valor** | Avisar antes da chuva forte e mostrar, com relatos confirmados por foto, onde a água já tomou a rua |
| **Componente desta disciplina** | Visão: transforma a foto ou o vídeo da rua em um nível de risco confiável |
| **Arquitetura: componentes** | Pipeline de dados (Airflow → Databricks, camada Ouro) · **Visão (YOLO → ONNX no navegador)** · RAG de orientações (GenAI) · Automação de alertas (RPA) · Sensor de chuva (Maker) · Deploy e monitoramento (DevOps) |
| **Fontes de dados** | Open-Meteo, IBGE, relatos de cidadãos · **Visão:** datasets abertos do Roboflow Universe (CC BY 4.0) |
| **Modelos de IA utilizados** | **YOLO11n e YOLO26n (Ultralytics)**, *fine-tuning* a partir de pesos COCO; a melhor arquitetura é escolhida na validação |
| **URL do MVP e repositório** | ver links acima |
| **Indicadores de impacto** | % de relatos com foto confirmados; tempo entre o primeiro relato e o alerta no mapa; recall de alerta (nível ≥ Alto) no teste |
| **ODS conectados** | 11 (cidades sustentáveis, meta 11.5: reduzir mortes e perdas por desastres), 13 (adaptação a eventos climáticos, meta 13.1), 9 (infraestrutura resiliente) |

## Como o componente se encaixa no produto

```
Cidadão (foto ou câmera)                Câmera de rua (vídeo / RTSP)
          │                                        │
          ▼                                        ▼
  web/ (navegador, ONNX)                 src/detectar.py (tempo real)
          │  a foto NÃO sai do aparelho            │  só eventos, sem imagem
          ▼                                        ▼
  Regra de risco (src/risco.py = web/detector.js) + confirmação humana
          │
          ▼
  Relato JSON {municipio, nivel_agua, classes, cobertura, horario}
          │
          ▼
  Pipeline de dados (relatos → Bronze/Prata/Ouro) → mapa de alertas do MVP
```

## Estrutura do repositório

```
notebooks/treino_colab.ipynb   treino completo no Google Colab (GPU T4)
src/classes.py                 3 classes do projeto + sinônimos dos datasets públicos
src/preparar_dataset.py        une datasets do Roboflow, traduz classes, remove duplicadas
src/risco.py                   regra de risco + filtro temporal para vídeo
src/detectar.py                detecção em tempo real (webcam, vídeo, RTSP, fotos)
src/avaliar.py                 métricas no TESTE: mAP, precisão, recall, acerto do nível de risco, FPS
src/exportar.py                exporta para ONNX e copia para web/model/
web/                           página que roda o modelo no navegador (onnxruntime-web)
docs/resultados/               métricas e gráficos gerados no treino
docs/dataset_stats.md          tamanho do dataset por split e por classe
tests/                         testes da regra de risco e um gerador de dados sintéticos
```

## Como reproduzir

### 1. Treinar (Google Colab, ~30 min com GPU T4)
1. Crie uma conta gratuita no [Roboflow](https://roboflow.com) e copie a sua Private API KEYS (Settings -> API Keys).
2. Abra `notebooks/treino_colab.ipynb` no Colab, mude o ambiente para **GPU T4** e, em **Secrets**, crie `ROBOFLOW_API_KEY`.
3. Na célula 2, troque `REPO_URL` pela URL deste repositório. **Executar tudo.**
4. No fim, o notebook baixa `resultados_visao.zip`. Descompacte na raiz do repositório (cria `modelos/`, `docs/resultados/` e `web/model/`).

### 2. Rodar em tempo real no computador
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python src/detectar.py --fonte 0                                   # webcam
python src/detectar.py --fonte amostras/video.mp4 --salvar saida/demo.mp4 --eventos saida/eventos.jsonl --municipio "Sao Paulo"
python src/detectar.py --fonte amostras/fotos/                      # pasta de fotos
```
Aperte **q** para fechar a janela.

### 3. Rodar a página web localmente
```bash
python -m http.server 8000
```
Abra `http://localhost:8000/web/`. A câmera do navegador só funciona em `localhost` ou `https` (GitHub Pages).

### 4. Testes
```bash
python -m pytest -q
```

## Regra de risco (IA explicável)

O modelo só diz **o que** aparece na imagem e **onde**. O nível de risco sai de uma regra simples e auditável, igual no Python e no navegador:

| Condição (confiança ≥ 35%) | Nível | Orientação ao cidadão |
|---|---|---|
| pessoa na água | **Crítico** | Ligue 193 / 199 |
| veículo ilhado, ou água em ≥ 35% da imagem | **Alto** | Via possivelmente intransitável, procure outra rota |
| água em ≥ 5% da imagem | **Atenção** | Reduza a velocidade, evite poças fundas |
| nada acima do limite | **Sem alagamento** | — |

No vídeo, um nível só aparece se estiver presente em pelo menos 6 dos últimos 15 quadros. Isso evita alarme falso por causa de um único quadro errado.

## Segurança, LGPD e IA responsável
- **Foto não sai do aparelho:** no navegador, a inferência roda localmente (WebAssembly). O relato leva só o município, o nível, as classes e o horário.
- **Sem dados pessoais nos eventos:** o `detectar.py` não grava imagens, a menos que `--salvar` seja usado. O arquivo de eventos guarda só os metadados.
- **Pessoa no circuito:** a IA sugere o nível e o cidadão confirma ou corrige. A correção fica registrada (`corrigido_pelo_cidadao`) e serve para medir e retreinar o modelo. Nos níveis Alto e Crítico vindos de câmera, o evento sai marcado com `requer_confirmacao_humana`.
- **Credenciais fora do código:** a chave do Roboflow fica nos *Secrets* do Colab.
- **Limitações conhecidas:** desempenho menor à noite, com gotas na lente e em reflexos de asfalto molhado. Os datasets públicos têm poucas cenas brasileiras, o que é um viés de domínio. A mitigação é acrescentar fotos negativas de ruas de SP (opção no notebook) e retreinar com os relatos corrigidos.

## Créditos dos dados
Datasets do Roboflow Universe sob licença CC BY 4.0. As referências completas, com autor, link e versão, estão em `docs/creditos_datasets.md` (gerado pelo notebook) e no relatório PDF.
