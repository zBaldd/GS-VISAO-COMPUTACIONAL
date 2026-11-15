"""Detecção de alagamento em tempo real (webcam, vídeo, câmera IP ou imagens).

Exemplos:
  python src/detectar.py --fonte 0                         # webcam
  python src/detectar.py --fonte amostras/video_sp.mp4 --salvar saida/demo.mp4
  python src/detectar.py --fonte rtsp://usuario:senha@ip/stream
  python src/detectar.py --fonte amostras/fotos/            # pasta de imagens

O que aparece na tela: caixas com a classe em português, a faixa de risco
(SEM ALAGAMENTO / ATENCAO / ALTO / CRITICO), a cobertura de água e o FPS.

LGPD: nenhuma imagem é gravada, a não ser que você peça (--salvar). O arquivo de
eventos (--eventos) guarda só horário, nível, cobertura e classes — sem imagem.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES, ROTULOS_PT  
from risco import NIVEIS, Deteccao, SuavizadorRisco, nivel_da_imagem  

CORES_CLASSE = {"area_alagada": (230, 140, 30), "veiculo_ilhado": (0, 165, 255), "pessoa_na_agua": (40, 40, 220)}
CORES_NIVEL = {0: (76, 160, 31), 1: (0, 200, 255), 2: (0, 110, 240), 3: (40, 40, 210)}
EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extrair(resultado):
    """Converte o resultado do ultralytics em uma lista de Deteccao normalizada."""
    dets = []
    nomes = resultado.names
    for caixa, conf, cls in zip(resultado.boxes.xyxyn.tolist(), resultado.boxes.conf.tolist(),
                                resultado.boxes.cls.tolist()):
        nome = nomes[int(cls)]
        if nome in CLASSES:
            dets.append(Deteccao(nome, float(conf), *caixa))
    return dets


def desenhar(frame, dets, nivel, cobertura, fps=None):
    h, w = frame.shape[:2]
    esp = max(2, round(w / 500))
    fonte_cx = max(0.35, w / 2000)
    for d in sorted(dets, key=lambda d: d.classe != "area_alagada"):  
        cor = CORES_CLASSE[d.classe]
        p1, p2 = (int(d.x1 * w), int(d.y1 * h)), (int(d.x2 * w), int(d.y2 * h))
        cv2.rectangle(frame, p1, p2, cor, esp)
        txt = f"{ROTULOS_PT[d.classe]} {d.conf:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fonte_cx, 1)
        cv2.rectangle(frame, (p1[0], max(0, p1[1] - th - 8)), (p1[0] + tw + 6, p1[1]), cor, -1)
        cv2.putText(frame, txt, (p1[0] + 3, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, fonte_cx,
                    (255, 255, 255), 1, cv2.LINE_AA)
    titulo, orientacao = NIVEIS[nivel]
    escala = max(0.35, w / 1400)            
    faixa = int(80 * escala) + 12
    cv2.rectangle(frame, (0, 0), (w, faixa), CORES_NIVEL[nivel], -1)
    cv2.putText(frame, f"RISCO: {titulo}  |  agua {cobertura:.0%}" + (f"  |  {fps:.0f} FPS" if fps else ""),
                (10, int(faixa * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255),
                max(1, int(2 * escala)), cv2.LINE_AA)
    cv2.putText(frame, orientacao, (10, int(faixa * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, escala * 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def evento(nivel, cobertura, dets, fonte, municipio):
    titulo, _ = NIVEIS[nivel]
    return {
        "data_hora": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fonte": "camera" if str(fonte).isdigit() or str(fonte).startswith("rtsp") else "arquivo",
        "municipio": municipio,
        "nivel": nivel,
        "nivel_texto": titulo,
        "cobertura_agua": round(cobertura, 3),
        "classes": sorted({d.classe for d in dets}),
        "origem": "visao_computacional",
        "requer_confirmacao_humana": nivel >= 2,
    }


def rodar_imagens(modelo, pasta, args):
    fotos = [pasta] if pasta.is_file() else sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXT_IMG)
    saida = Path(args.salvar or "saida/imagens")
    saida.mkdir(parents=True, exist_ok=True)
    for foto in fotos:
        r = modelo.predict(str(foto), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dets = extrair(r)
        nivel, cob, motivo = nivel_da_imagem(dets)
        img = desenhar(r.orig_img.copy(), dets, nivel, cob)
        cv2.imwrite(str(saida / foto.name), img)
        print(f"{foto.name}: {NIVEIS[nivel][0]} ({motivo})")
    print(f"Imagens anotadas em {saida}/")


def rodar_video(modelo, args):
    fonte = int(args.fonte) if str(args.fonte).isdigit() else args.fonte
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir a fonte {args.fonte}")
    gravador = None
    suav = SuavizadorRisco(janela=args.janela, min_quadros=args.min_quadros)
    nivel_anterior, t_ant, fps = -1, time.time(), 0.0
    arq_eventos = None
    if args.eventos:
        Path(args.eventos).parent.mkdir(parents=True, exist_ok=True)
        arq_eventos = open(args.eventos, "a", encoding="utf-8")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = modelo.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dets = extrair(r)
        nivel_q, cob, _ = nivel_da_imagem(dets)
        nivel = suav.atualizar(nivel_q)
        agora = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(agora - t_ant, 1e-6)) if fps else 1 / max(agora - t_ant, 1e-6)
        t_ant = agora
        desenhar(frame, dets, nivel, cob, fps)
        if nivel != nivel_anterior:
            ev = evento(nivel, cob, dets, args.fonte, args.municipio)
            print(json.dumps(ev, ensure_ascii=False))
            if arq_eventos:
                arq_eventos.write(json.dumps(ev, ensure_ascii=False) + "\n")
                arq_eventos.flush()
            nivel_anterior = nivel
        if args.salvar:
            if gravador is None:
                Path(args.salvar).parent.mkdir(parents=True, exist_ok=True)
                fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
                gravador = cv2.VideoWriter(args.salvar, cv2.VideoWriter_fourcc(*"mp4v"), fps_in,
                                           (frame.shape[1], frame.shape[0]))
            gravador.write(frame)
        if not args.sem_janela:
            cv2.imshow("Alerta Chuva SP - Visao (q para sair)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if gravador:
        gravador.release()
        print(f"Vídeo salvo em {args.salvar}")
    if arq_eventos:
        arq_eventos.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="modelos/best.pt", help=".pt ou .onnx treinado")
    ap.add_argument("--fonte", default="0", help="0 = webcam, caminho de vídeo/imagem/pasta ou URL rtsp")
    ap.add_argument("--conf", type=float, default=0.25, help="confiança mínima exibida")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--salvar", help="caminho do vídeo (ou pasta, para imagens) de saída")
    ap.add_argument("--eventos", help="arquivo .jsonl para registrar mudanças de nível")
    ap.add_argument("--municipio", default=None, help="município da câmera, ex.: 'Sao Paulo'")
    ap.add_argument("--janela", type=int, default=15)
    ap.add_argument("--min-quadros", dest="min_quadros", type=int, default=6)
    ap.add_argument("--sem-janela", dest="sem_janela", action="store_true", help="não abre janela (servidor)")
    a = ap.parse_args()
    m = YOLO(a.modelo, task="detect")
    caminho = Path(a.fonte)
    if caminho.exists() and (caminho.is_dir() or caminho.suffix.lower() in EXT_IMG):
        rodar_imagens(m, caminho, a)
    else:
        rodar_video(m, a)
