"""Une um ou mais datasets exportados do Roboflow (formato YOLOv8/YOLO11) em um
único dataset com as 3 classes do projeto.

O que o script faz:
  1. Lê o data.yaml de cada export e traduz os nomes de classe com classes.SINONIMOS
     (ou com um mapa próprio do dataset, quando o mesmo nome significa outra coisa,
     ex.: "car" num dataset de enchente = veículo ilhado).
  2. Descarta classes que não interessam ao projeto (ex.: "house", "tree").
  3. Converte polígonos (exports de segmentação) em caixas.
  4. Remove imagens duplicadas entre datasets (hash MD5).
  5. Opcional (--resplit): refaz treino/validação/teste juntando tudo, separando por
     IMAGEM ORIGINAL (as cópias aumentadas do Roboflow, "foto_jpg.rf.xxx.jpg", ficam
     sempre no mesmo split) e estratificando pelas classes raras, para o teste ter
     pessoas e veículos suficientes. Validação e teste ficam só com 1 cópia de cada foto.
  6. Opcional: limita o número de fotos de um dataset muito grande (max_imagens).
  7. Opcional: acrescenta fotos "negativas" (ruas secas) sem rótulo.
  8. Lista em sem_agua.txt as imagens de datasets que NÃO rotulam água, para a
     pseudo-rotulagem (src/pseudo_rotular.py).
  9. Gera data.yaml e estatísticas (docs/dataset_stats.json e .md).

Uso simples (mantém os splits originais):
  python src/preparar_dataset.py --entradas datasets/a datasets/b --saida dados/alagamento

Uso com configuração por dataset e novo split:
  python src/preparar_dataset.py --config datasets.json --saida dados/alagamento --resplit 0.75,0.125,0.125
  datasets.json = [{"pasta": "datasets/flood-cka5z", "mapa": {"car": "veiculo_ilhado", "person": "pessoa_na_agua"},
                    "max_imagens": 1500, "sem_agua": true}, ...]
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
RARAS = ["pessoa_na_agua", "veiculo_ilhado", "area_alagada"]  # ordem de prioridade na estratificação


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


def original_roboflow(nome_arquivo: str) -> str:
    """'rua_123_jpg.rf.ab12.jpg' -> 'rua_123_jpg' (todas as cópias aumentadas têm o mesmo prefixo)."""
    return nome_arquivo.split(".rf.")[0]


def tier(classes_presentes) -> int:
    for i, c in enumerate(RARAS):
        if c in classes_presentes:
            return i
    return len(RARAS)  # negativa


def ler_config(args):
    if args.config:
        itens = json.loads(Path(args.config).read_text(encoding="utf-8"))
    else:
        itens = [{"pasta": p} for p in args.entradas]
    for it in itens:
        it.setdefault("nome", Path(it["pasta"]).name)
        it.setdefault("mapa", {})
        it.setdefault("max_imagens", None)
        it.setdefault("sem_agua", False)
    return itens


def coletar(itens):
    """Lê todos os exports e devolve uma lista de registros (sem copiar nada ainda)."""
    vistos, registros = set(), []
    mapa_usado, descartadas = defaultdict(set), Counter()
    duplicadas = 0
    for i, it in enumerate(itens):
        pasta = Path(it["pasta"])
        nomes = nomes_do_yaml(pasta)
        mapa_local = {str(k).strip().lower(): v for k, v in it["mapa"].items()}
        traducao = {}
        for idx, nome in enumerate(nomes):
            chave = str(nome).strip().lower()
            alvo = mapa_local[chave] if chave in mapa_local else classe_do_projeto(nome)
            if alvo:
                traducao[idx] = CLASSES.index(alvo)
                mapa_usado[alvo].add(f"{it['nome']}:{nome}")
            else:
                descartadas[f"{it['nome']}:{nome}"] += 0
        doc = []
        for pasta_split in sorted(pasta.iterdir()):
            split = SPLITS.get(pasta_split.name)
            if not split or not (pasta_split / "images").exists():
                continue
            for img in sorted((pasta_split / "images").iterdir()):
                if img.suffix.lower() not in EXT_IMG:
                    continue
                h = md5(img)
                if h in vistos:
                    duplicadas += 1
                    continue
                vistos.add(h)
                linhas, presentes = [], set()
                rot = pasta_split / "labels" / (img.stem + ".txt")
                if rot.exists():
                    for ln in rot.read_text().splitlines():
                        p = ln.split()
                        if not p:
                            continue
                        idx = int(float(p[0]))
                        if idx not in traducao:
                            descartadas[f"{it['nome']}:{nomes[idx] if idx < len(nomes) else idx}"] += 1
                            continue
                        caixa = linha_para_caixa(p)
                        if not caixa or caixa[2] <= 0 or caixa[3] <= 0:
                            continue
                        linhas.append(f"{traducao[idx]} " + " ".join(f"{v:.6f}" for v in caixa))
                        presentes.add(CLASSES[traducao[idx]])
                doc.append({"img": img, "hash": h, "linhas": linhas, "presentes": presentes,
                            "split_orig": split, "grupo": f"d{i}:{original_roboflow(img.name)}",
                            "ds": i, "sem_agua": it["sem_agua"]})
        # limite de fotos por dataset (conta fotos originais, não cópias aumentadas)
        if it["max_imagens"]:
            grupos = defaultdict(list)
            for r in doc:
                grupos[r["grupo"]].append(r)
            rnd = random.Random(1000 + i)
            ordem = list(grupos)
            rnd.shuffle(ordem)
            ordem.sort(key=lambda g: min(tier(r["presentes"]) for r in grupos[g]))  # raras primeiro
            doc = [r for g in ordem[: it["max_imagens"]] for r in grupos[g]]
        registros += doc
    return registros, mapa_usado, descartadas, duplicadas


def refazer_split(registros, props, semente):
    """Split por grupo (foto original), estratificado pela classe mais rara de cada grupo."""
    grupos = defaultdict(list)
    for r in registros:
        grupos[r["grupo"]].append(r)
    por_tier = defaultdict(list)
    for g, rs in grupos.items():
        presentes = set().union(*(r["presentes"] for r in rs))
        por_tier[tier(presentes)].append(g)
    rnd = random.Random(semente)
    nomes = ["train", "valid", "test"]
    destino = {}
    for t in sorted(por_tier):
        gs = sorted(por_tier[t])
        rnd.shuffle(gs)
        cont = [0, 0, 0]
        for k, g in enumerate(gs):
            s = max(range(3), key=lambda j: props[j] * (k + 1) - cont[j])  # maior "atraso" em relação à meta
            cont[s] += 1
            destino[g] = nomes[s]
    saida = []
    for g, rs in grupos.items():
        s = destino[g]
        if s == "train":
            saida += [(r, s) for r in rs]
        else:
            saida.append((sorted(rs, key=lambda r: r["img"].name)[0], s))  # 1 cópia só em validação/teste
    return saida


def preparar(itens, saida: Path, negativos=None, resplit=None, semente=42):
    if saida.exists():
        shutil.rmtree(saida)
    for s in ("train", "valid", "test"):
        (saida / s / "images").mkdir(parents=True)
        (saida / s / "labels").mkdir(parents=True)

    registros, mapa_usado, descartadas, duplicadas = coletar(itens)
    if resplit:
        alocados = refazer_split(registros, resplit, semente)
    else:
        alocados = [(r, r["split_orig"]) for r in registros]
        if not any(s == "test" for _, s in alocados):  # export sem teste: metade da validação vira teste
            val = [k for k, (_, s) in enumerate(alocados) if s == "valid"]
            random.Random(semente).shuffle(val)
            for k in val[: len(val) // 2]:
                alocados[k] = (alocados[k][0], "test")

    stats = {s: {"imagens": 0, "imagens_sem_rotulo": 0, "instancias": Counter()} for s in ("train", "valid", "test")}
    sem_agua = []
    for r, split in alocados:
        novo = f"d{r['ds']}_{r['hash'][:12]}"
        nome_img = novo + r["img"].suffix.lower()
        shutil.copy2(r["img"], saida / split / "images" / nome_img)
        (saida / split / "labels" / (novo + ".txt")).write_text("\n".join(r["linhas"]))
        stats[split]["imagens"] += 1
        stats[split]["imagens_sem_rotulo"] += int(not r["linhas"])
        for ln in r["linhas"]:
            stats[split]["instancias"][CLASSES[int(ln.split()[0])]] += 1
        if r["sem_agua"]:
            sem_agua.append(f"{split}/images/{nome_img}")

    # fotos negativas (ruas secas, sem alagamento): 80/10/10
    if negativos:
        rnd = random.Random(semente)
        fotos = sorted(p for p in Path(negativos).rglob("*") if p.suffix.lower() in EXT_IMG)
        rnd.shuffle(fotos)
        for k, img in enumerate(fotos):
            split = "train" if k % 10 < 8 else ("valid" if k % 10 == 8 else "test")
            novo = f"neg_{md5(img)[:12]}"
            shutil.copy2(img, saida / split / "images" / (novo + img.suffix.lower()))
            (saida / split / "labels" / (novo + ".txt")).write_text("")
            stats[split]["imagens"] += 1
            stats[split]["imagens_sem_rotulo"] += 1

    (saida / "sem_agua.txt").write_text("\n".join(sem_agua))
    (saida / "data.yaml").write_text(
        yaml.safe_dump({"path": str(saida.resolve()), "train": "train/images", "val": "valid/images",
                        "test": "test/images", "nc": len(CLASSES), "names": CLASSES},
                       allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    return {
        "classes": CLASSES,
        "datasets": [{"nome": it["nome"], "mapa_proprio": it["mapa"], "max_imagens": it["max_imagens"],
                      "sem_agua": it["sem_agua"]} for it in itens],
        "split": "refeito por foto original, estratificado" if resplit else "original dos exports",
        "proporcoes": resplit,
        "origem_das_classes": {k: sorted(v) for k, v in mapa_usado.items()},
        "classes_descartadas": sorted(descartadas),
        "duplicadas_removidas": duplicadas,
        "imagens_para_pseudo_rotulo": len(sem_agua),
        "splits": {s: {"imagens": v["imagens"], "imagens_sem_rotulo": v["imagens_sem_rotulo"],
                       "instancias": {c: v["instancias"].get(c, 0) for c in CLASSES}}
                   for s, v in stats.items()},
    }


def salvar_stats(resumo, pasta_docs: Path, sufixo=""):
    pasta_docs.mkdir(parents=True, exist_ok=True)
    (pasta_docs / f"dataset_stats{sufixo}.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                                            encoding="utf-8")
    linhas = ["| Split | Imagens | Sem rótulo (negativas) | " + " | ".join(CLASSES) + " |",
              "|---|---|---|" + "---|" * len(CLASSES)]
    for s, v in resumo["splits"].items():
        linhas.append(f"| {s} | {v['imagens']} | {v['imagens_sem_rotulo']} | "
                      + " | ".join(str(v["instancias"][c]) for c in CLASSES) + " |")
    linhas += ["", f"Split: {resumo['split']}", f"Duplicadas removidas: {resumo['duplicadas_removidas']}",
               f"Imagens enviadas para pseudo-rotulagem de água: {resumo['imagens_para_pseudo_rotulo']}",
               "", "Origem das classes:"]
    for c, origem in resumo["origem_das_classes"].items():
        linhas.append(f"- **{c}** ← {', '.join(origem)}")
    if resumo["classes_descartadas"]:
        linhas.append(f"\nClasses descartadas: {', '.join(resumo['classes_descartadas'])}")
    (pasta_docs / f"dataset_stats{sufixo}.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--entradas", nargs="+", help="pastas de exports YOLO do Roboflow")
    g.add_argument("--config", help="JSON com a lista de datasets e opções por dataset")
    ap.add_argument("--saida", default="dados/alagamento")
    ap.add_argument("--negativos", help="pasta com fotos de ruas secas (sem rótulo)")
    ap.add_argument("--resplit", help="refaz o split, ex.: 0.75,0.125,0.125")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--sufixo", default="", help="sufixo nos arquivos de estatística, ex.: _v2")
    a = ap.parse_args()
    props = [float(x) for x in a.resplit.split(",")] if a.resplit else None
    r = preparar(ler_config(a), Path(a.saida), a.negativos, props)
    salvar_stats(r, Path(a.docs), a.sufixo)
