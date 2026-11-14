"""Gera um mini "export do Roboflow" sintético para testar o pipeline sem internet.
NÃO serve para treinar o modelo real — só para validar os scripts.

Uso: python tests/gerar_sintetico.py /tmp/sintetico
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

NOMES = ["flooded_area", "flooded_car", "flooded_person", "house"]  # nomes "estrangeiros" de propósito


def imagem(rnd, n=320):
    img = np.full((n, n, 3), rnd.randint(90, 140), np.uint8)
    img[: n // 3] = (200, 170, 120)  # céu
    rot = []
    if rnd.random() < 0.8:
        y1 = rnd.randint(n // 3, n // 2)
        x1, x2 = rnd.randint(0, n // 4), rnd.randint(n * 3 // 4, n)
        cv2.rectangle(img, (x1, y1), (x2, n - 1), (150, 110, 60), -1)  # água
        rot.append((0, x1, y1, x2, n - 1))
        if rnd.random() < 0.5:
            cx, cy = rnd.randint(x1 + 30, x2 - 60), rnd.randint(y1, n - 40)
            cv2.rectangle(img, (cx, cy), (cx + 50, cy + 25), (30, 30, 200), -1)  # carro
            rot.append((1, cx, cy, cx + 50, cy + 25))
        if rnd.random() < 0.3:
            px, py = rnd.randint(x1 + 10, x2 - 20), rnd.randint(y1, n - 40)
            cv2.circle(img, (px + 6, py + 6), 6, (60, 200, 230), -1)  # pessoa
            cv2.rectangle(img, (px, py + 12), (px + 12, py + 35), (60, 200, 230), -1)
            rot.append((2, px, py, px + 12, py + 35))
    hx = rnd.randint(0, n - 60)
    cv2.rectangle(img, (hx, n // 3 - 50), (hx + 60, n // 3), (80, 80, 80), -1)  # casa
    rot.append((3, hx, n // 3 - 50, hx + 60, n // 3))
    linhas = [f"{c} {(a+b)/2/n:.5f} {(d+e)/2/n:.5f} {(b-a)/n:.5f} {(e-d)/n:.5f}" for c, a, d, b, e in rot]
    return img, linhas


def main(destino):
    rnd = random.Random(0)
    d = Path(destino)
    for split, qtd in (("train", 60), ("valid", 16)):  # sem "test" de propósito
        (d / split / "images").mkdir(parents=True, exist_ok=True)
        (d / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(qtd):
            img, linhas = imagem(rnd)
            cv2.imwrite(str(d / split / "images" / f"s{split}{i}.jpg"), img)
            (d / split / "labels" / f"s{split}{i}.txt").write_text("\n".join(linhas))
    (d / "data.yaml").write_text(yaml.safe_dump({"names": NOMES, "nc": len(NOMES)}))
    # vídeo de 3 s para testar o modo tempo real
    vw = cv2.VideoWriter(str(d / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 320))
    for _ in range(30):
        vw.write(imagem(rnd)[0])
    vw.release()
    print("ok", d)


if __name__ == "__main__":
    main(sys.argv[1])
