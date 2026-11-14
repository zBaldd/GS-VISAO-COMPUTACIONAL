"""Une um ou mais datasets exportados do Roboflow (formato YOLOv8/YOLO11) em um
único dataset com as 3 classes do projeto.

O que o script faz:
  1. Lê o data.yaml de cada export e traduz os nomes de classe com classes.SINONIMOS.
  2. Descarta classes que não interessam ao projeto (ex.: "house", "tree").
  3. Converte polígonos (exports de segmentação) em caixas.
  4. Remove imagens duplicadas entre datasets (hash MD5), evitando vazamento
     de imagens iguais entre treino e teste.
  5. Opcional: acrescenta fotos "negativas" (ruas secas) sem rótulo, para o modelo
     aprender o que NÃO é alagamento.
  6. Gera data.yaml e estatísticas (docs/dataset_stats.json e .md).

Uso:
  python src/preparar_dataset.py --entradas datasets/roboflow1 datasets/roboflow2 \
      --saida dados/alagamento [--negativos amostras/ruas_secas]
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES, classe_do_projeto  # noqa: E402

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def md5(caminho: Path) -> str:
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def nomes_do_yaml(pasta: Path):
    yml = pasta / "data.yaml"
    if not yml.exists():
        raise FileNotFoundError(f"{yml} não encontrado. A pasta é um export YOLO do Roboflow?")
    dados = yaml.safe_load(yml.read_text(encoding="utf-8"))
    nomes = dados["names"]
    if isinstance(nomes, dict):
        nomes = [nomes[k] for k in sorted(nomes, key=int)]
    return nomes


def linha_para_caixa(partes):
    """Aceita 'cls cx cy w h' ou polígono 'cls x1 y1 x2 y2 ...'. Devolve (cx, cy, w, h)."""
    vals = [float(v) for v in partes[1:]]
    if len(vals) == 4:
        return tuple(vals)
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return None


def preparar(entradas, saida: Path, negativos=None, semente=42):
    if saida.exists():
        shutil.rmtree(saida)
    for s in ("train", "valid", "test"):
        (saida / s / "images").mkdir(parents=True)
        (saida / s / "labels").mkdir(parents=True)

    vistos = set()
    stats = {s: {"imagens": 0, "imagens_sem_rotulo": 0, "instancias": Counter()} for s in ("train", "valid", "test")}
    mapa_usado = defaultdict(set)
    descartadas = Counter()
    duplicadas = 0
    tem_teste = False

    for i, pasta in enumerate(map(Path, entradas)):
        nomes = nomes_do_yaml(pasta)
        traducao = {}
        for idx, nome in enumerate(nomes):
            alvo = classe_do_projeto(nome)
            if alvo:
                traducao[idx] = CLASSES.index(alvo)
                mapa_usado[alvo].add(f"{pasta.name}:{nome}")
            else:
                descartadas[f"{pasta.name}:{nome}"] += 0
        for pasta_split in pasta.iterdir():
            split = SPLITS.get(pasta_split.name)
            if not split or not (pasta_split / "images").exists():
                continue
            tem_teste |= split == "test"
            for img in sorted((pasta_split / "images").iterdir()):
                if img.suffix.lower() not in EXT_IMG:
                    continue
                h = md5(img)
                if h in vistos:
                    duplicadas += 1
                    continue
                vistos.add(h)
                linhas = []
                rot = pasta_split / "labels" / (img.stem + ".txt")
                if rot.exists():
                    for ln in rot.read_text().splitlines():
                        p = ln.split()
                        if not p:
                            continue
                        idx = int(float(p[0]))
                        if idx not in traducao:
                            descartadas[f"{pasta.name}:{nomes[idx]}"] += 1
                            continue
                        caixa = linha_para_caixa(p)
                        if not caixa or caixa[2] <= 0 or caixa[3] <= 0:
                            continue
                        linhas.append(f"{traducao[idx]} " + " ".join(f"{v:.6f}" for v in caixa))
                        stats[split]["instancias"][CLASSES[traducao[idx]]] += 1
                novo = f"d{i}_{h[:12]}"
                shutil.copy2(img, saida / split / "images" / (novo + img.suffix.lower()))
                (saida / split / "labels" / (novo + ".txt")).write_text("\n".join(linhas))
                stats[split]["imagens"] += 1
                stats[split]["imagens_sem_rotulo"] += int(not linhas)

    rnd = random.Random(semente)
    # sem pasta de teste nos exports: separa metade da validação para teste
    if not tem_teste:
        imgs_val = sorted((saida / "valid" / "images").iterdir())
        rnd.shuffle(imgs_val)
        for img in imgs_val[: len(imgs_val) // 2]:
            lbl = saida / "valid" / "labels" / (img.stem + ".txt")
            linhas = lbl.read_text().splitlines()
            shutil.move(img, saida / "test" / "images" / img.name)
            shutil.move(lbl, saida / "test" / "labels" / lbl.name)
            stats["valid"]["imagens"] -= 1
            stats["test"]["imagens"] += 1
            if not linhas:
                stats["valid"]["imagens_sem_rotulo"] -= 1
                stats["test"]["imagens_sem_rotulo"] += 1
            for ln in linhas:
                c = CLASSES[int(ln.split()[0])]
                stats["valid"]["instancias"][c] -= 1
                stats["test"]["instancias"][c] += 1

    # fotos negativas (ruas secas, sem alagamento): 80/10/10
    if negativos:
        fotos = sorted(p for p in Path(negativos).rglob("*") if p.suffix.lower() in EXT_IMG)
        rnd.shuffle(fotos)
        for k, img in enumerate(fotos):
            h = md5(img)
            if h in vistos:
                continue
            vistos.add(h)
            split = "train" if k % 10 < 8 else ("valid" if k % 10 == 8 else "test")
            novo = f"neg_{h[:12]}"
            shutil.copy2(img, saida / split / "images" / (novo + img.suffix.lower()))
            (saida / split / "labels" / (novo + ".txt")).write_text("")
            stats[split]["imagens"] += 1
            stats[split]["imagens_sem_rotulo"] += 1

    (saida / "data.yaml").write_text(
        yaml.safe_dump({"path": str(saida.resolve()), "train": "train/images", "val": "valid/images",
                        "test": "test/images", "nc": len(CLASSES), "names": CLASSES},
                       allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    resumo = {
        "classes": CLASSES,
        "origem_das_classes": {k: sorted(v) for k, v in mapa_usado.items()},
        "classes_descartadas": sorted(descartadas),
        "duplicadas_removidas": duplicadas,
        "splits": {s: {"imagens": v["imagens"], "imagens_sem_rotulo": v["imagens_sem_rotulo"],
                       "instancias": {c: v["instancias"].get(c, 0) for c in CLASSES}}
                   for s, v in stats.items()},
    }
    return resumo


def salvar_stats(resumo, pasta_docs: Path):
    pasta_docs.mkdir(parents=True, exist_ok=True)
    (pasta_docs / "dataset_stats.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    linhas = ["| Split | Imagens | Sem rótulo (negativas) | " + " | ".join(CLASSES) + " |",
              "|---|---|---|" + "---|" * len(CLASSES)]
    for s, v in resumo["splits"].items():
        linhas.append(f"| {s} | {v['imagens']} | {v['imagens_sem_rotulo']} | "
                      + " | ".join(str(v["instancias"][c]) for c in CLASSES) + " |")
    linhas += ["", f"Duplicadas removidas: {resumo['duplicadas_removidas']}", "",
               "Origem das classes:"]
    for c, origem in resumo["origem_das_classes"].items():
        linhas.append(f"- **{c}** ← {', '.join(origem)}")
    if resumo["classes_descartadas"]:
        linhas.append(f"\nClasses descartadas: {', '.join(resumo['classes_descartadas'])}")
    (pasta_docs / "dataset_stats.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entradas", nargs="+", required=True, help="pastas de exports YOLO do Roboflow")
    ap.add_argument("--saida", default="dados/alagamento")
    ap.add_argument("--negativos", help="pasta com fotos de ruas secas (sem rótulo)")
    ap.add_argument("--docs", default="docs")
    a = ap.parse_args()
    r = preparar(a.entradas, Path(a.saida), a.negativos)
    salvar_stats(r, Path(a.docs))
