"""Exporta o modelo treinado para ONNX e copia para web/model/, onde a página
do MVP roda a detecção direto no navegador do cidadão (onnxruntime-web).

Uso:
  python src/exportar.py --modelo modelos/best.pt --imgsz 640
"""
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from ultralytics import YOLO


def main(a):
    modelo = YOLO(a.modelo)
    caminho = Path(modelo.export(format="onnx", imgsz=a.imgsz, opset=12, simplify=True, dynamic=False))
    destino = Path(a.web) / "model"
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copy2(caminho, destino / "alagamento.onnx")
    meta = {"classes": [modelo.names[i] for i in sorted(modelo.names)], "imgsz": a.imgsz,
            "exportado_em": str(date.today()), "origem": Path(a.modelo).name}
    met = Path(a.metricas)
    if met.exists():
        meta["metricas_teste"] = json.loads(met.read_text())
    (destino / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    mb = (destino / "alagamento.onnx").stat().st_size / 1e6
    print(f"ONNX copiado para {destino/'alagamento.onnx'} ({mb:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="modelos/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--web", default="web")
    ap.add_argument("--metricas", default="docs/resultados/metricas.json")
    main(ap.parse_args())
