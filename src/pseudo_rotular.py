"""Pseudo-rotulagem de água nas imagens de datasets que só rotulam carro e pessoa.

Problema: os datasets que trazem muitos veículos ilhados e pessoas na água NÃO
desenham caixa na água. Se entrarem assim no treino, o modelo aprende que "água
sem caixa = fundo" e piora justamente na classe que já era boa (area_alagada).

Solução (aprendizado semi-supervisionado): o modelo da versão 1, que já acerta
89% de mAP50 em area_alagada, rotula a água dessas imagens. Só entram caixas com
confiança alta (padrão 0,50), e só nas imagens listadas em sem_agua.txt (gerado
pelo preparar_dataset.py). Os rótulos humanos de carro e pessoa são mantidos.

Uso:
  python src/pseudo_rotular.py --dados dados/alagamento --modelo modelos/best.pt --conf 0.5
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES  # noqa: E402


def main(a):
    raiz = Path(a.dados)
    lista = [l for l in (raiz / "sem_agua.txt").read_text().splitlines() if l.strip()]
    alvo = CLASSES.index(a.classe)
    modelo = YOLO(a.modelo, task="detect")
    idx_modelo = [k for k, v in modelo.names.items() if v == a.classe]
    if not idx_modelo:
        sys.exit(f"O modelo {a.modelo} não tem a classe {a.classe}")
    stats = {s: Counter() for s in ("train", "valid", "test")}
    rotuladas = []
    for i in range(0, len(lista), a.lote):
        lote = lista[i:i + a.lote]
        resultados = modelo.predict([str(raiz / p) for p in lote], conf=a.conf, imgsz=a.imgsz,
                                    classes=idx_modelo, verbose=False)
        for rel, r in zip(lote, resultados):
            split = rel.split("/")[0]
            stats[split]["imagens"] += 1
            lbl = raiz / split / "labels" / (Path(rel).stem + ".txt")
            linhas = [l for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
            if any(int(l.split()[0]) == alvo for l in linhas):
                continue  # já tem água rotulada por humano
            novas = [f"{alvo} " + " ".join(f"{v:.6f}" for v in cx) for cx in r.boxes.xywhn.tolist()]
            if novas:
                lbl.write_text("\n".join(linhas + novas))
                stats[split]["imagens_com_pseudo"] += 1
                stats[split]["caixas"] += len(novas)
                rotuladas.append(rel)
    resumo = {"modelo": Path(a.modelo).name, "classe": a.classe, "conf_minima": a.conf,
              "por_split": {s: dict(c) for s, c in stats.items()}}
    Path(a.docs).mkdir(parents=True, exist_ok=True)
    (Path(a.docs) / "pseudo_rotulos.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    (raiz / "pseudo_rotuladas.txt").write_text("\n".join(rotuladas))
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", default="dados/alagamento")
    ap.add_argument("--modelo", default="modelos/best.pt")
    ap.add_argument("--classe", default="area_alagada")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--docs", default="docs")
    main(ap.parse_args())
