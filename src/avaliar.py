"""Avalia o modelo treinado no split de TESTE (imagens que o modelo nunca viu).

Gera em docs/resultados/:
  - metricas.json        mAP50, mAP50-95, precisão e recall (geral e por classe)
  - nivel_risco.json     acurácia do NÍVEL DE RISCO (a métrica que importa para o cidadão)
  - velocidade.json      tempo por imagem e FPS no hardware atual
  - confusion_matrix.png e curvas PR/F1 geradas pelo ultralytics

Uso:
  python src/avaliar.py --modelo modelos/best.pt --dados dados/alagamento/data.yaml
"""
import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES  
from risco import NIVEIS, Deteccao, nivel_da_imagem  
from detectar import extrair  


def gt_deteccoes(arq_rotulo: Path):
    dets = []
    if arq_rotulo.exists():
        for ln in arq_rotulo.read_text().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
            dets.append(Deteccao(CLASSES[c], 1.0, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return dets


def main(a):
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(a.modelo, task="detect")

    #1) métricas de detecção no teste
    m = modelo.val(data=a.dados, split="test", imgsz=a.imgsz, batch=a.batch, plots=True,
                   project=str(saida), name="val_teste", exist_ok=True, verbose=False)
    por_classe = {}
    for i, c in enumerate(CLASSES):
        if i in list(m.box.ap_class_index):
            j = list(m.box.ap_class_index).index(i)
            p, r, ap50, ap = m.box.class_result(j)
            por_classe[c] = {"precisao": round(float(p), 4), "recall": round(float(r), 4),
                             "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    metricas = {"split": "test", "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
                "precisao": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
                "por_classe": por_classe}
    (saida / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    for f in Path(m.save_dir).glob("*.png"):
        shutil.copy2(f, saida / f.name)

    #2) nível de risco: previsto pelo modelo x calculado a partir do rótulo humano
    cfg = yaml.safe_load(Path(a.dados).read_text(encoding="utf-8"))
    raiz = Path(cfg.get("path", Path(a.dados).parent))
    pasta_img = raiz / cfg["test"]
    pasta_lbl = pasta_img.parent / "labels"
    imgs = sorted(p for p in pasta_img.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    matriz = np.zeros((4, 4), dtype=int)
    tempos = []
    for img in imgs:
        t0 = time.perf_counter()
        r = modelo.predict(str(img), imgsz=a.imgsz, conf=0.25, verbose=False)[0]
        tempos.append(time.perf_counter() - t0)
        prev, _, _ = nivel_da_imagem(extrair(r))
        real, _, _ = nivel_da_imagem(gt_deteccoes(pasta_lbl / (img.stem + ".txt")))
        matriz[real, prev] += 1
    total = matriz.sum()
    acerto = int(np.trace(matriz))
    #"alerta" = nível >= 2 (via intransitável ou pessoa em risco)
    vp = int(matriz[2:, 2:].sum()); fn = int(matriz[2:, :2].sum()); fp = int(matriz[:2, 2:].sum())
    risco = {
        "imagens": int(total),
        "acuracia_nivel": round(acerto / total, 4) if total else None,
        "acuracia_nivel_tolerancia_1": round(sum(matriz[i, j] for i in range(4) for j in range(4)
                                                 if abs(i - j) <= 1) / total, 4) if total else None,
        "alerta_recall": round(vp / (vp + fn), 4) if vp + fn else None,
        "alerta_precisao": round(vp / (vp + fp), 4) if vp + fp else None,
        "niveis": [NIVEIS[i][0] for i in range(4)],
        "matriz_real_x_previsto": matriz.tolist(),
        "distribuicao_real": dict(Counter({NIVEIS[i][0]: int(matriz[i].sum()) for i in range(4)})),
    }
    (saida / "nivel_risco.json").write_text(json.dumps(risco, indent=2, ensure_ascii=False))

    #3) velocidade (descarta as 3 primeiras, que incluem aquecimento)
    t = np.array(tempos[3:] or tempos)
    import platform
    import torch
    vel = {"hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "CPU",
           "imgsz": a.imgsz, "ms_por_imagem_mediana": round(float(np.median(t) * 1000), 1),
           "fps_aprox": round(float(1 / np.median(t)), 1)}
    (saida / "velocidade.json").write_text(json.dumps(vel, indent=2, ensure_ascii=False))

    print(json.dumps({"deteccao": metricas, "nivel_risco": {k: risco[k] for k in
                      ("acuracia_nivel", "alerta_recall", "alerta_precisao")}, "velocidade": vel},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="modelos/best.pt")
    ap.add_argument("--dados", default="dados/alagamento/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--saida", default="docs/resultados")
    main(ap.parse_args())
