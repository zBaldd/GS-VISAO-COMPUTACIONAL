// Detector de alagamento no navegador (onnxruntime-web).
// Mesmas regras de src/risco.py — se mudar uma, mude a outra.
// Funciona com objetos "ImageData-like" ({data, width, height}, RGBA), então
// o mesmo código roda no navegador e nos testes em Node.

export const ROTULOS = {
  area_alagada: "Rua alagada",
  veiculo_ilhado: "Veículo ilhado",
  pessoa_na_agua: "Pessoa na água",
};

export const NIVEIS = [
  { titulo: "Sem alagamento", orientacao: "Nenhum sinal de alagamento na foto." },
  { titulo: "Atenção", orientacao: "Água na via. Reduza a velocidade e evite poças fundas." },
  { titulo: "Alto", orientacao: "Via possivelmente intransitável. Procure outra rota e não atravesse." },
  { titulo: "Crítico", orientacao: "Pessoa em risco na água. Ligue 193 (Bombeiros) ou 199 (Defesa Civil)." },
];

export const REGRAS = {
  CONF_MIN: 0.35,          // confiança mínima para entrar na regra de risco
  COBERTURA_ATENCAO: 0.05, // >= 5% da imagem com água
  COBERTURA_ALTO: 0.35,    // >= 35% da imagem com água
  GRADE: 64,
};

// ---------- pré-processamento: letterbox para imgsz x imgsz, RGB, 0..1, CHW ----------
export function preprocessar(img, imgsz) {
  const r = Math.min(imgsz / img.width, imgsz / img.height);
  const nw = Math.round(img.width * r), nh = Math.round(img.height * r);
  const padX = (imgsz - nw) / 2, padY = (imgsz - nh) / 2;
  const left = Math.round(padX - 0.1), top = Math.round(padY - 0.1);
  const area = imgsz * imgsz;
  const t = new Float32Array(3 * area).fill(114 / 255);
  // redimensionamento bilinear simples
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(img.height - 1, (y + 0.5) / r - 0.5);
    const y0 = Math.max(0, Math.floor(sy)), y1 = Math.min(img.height - 1, y0 + 1), fy = Math.max(0, sy - y0);
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(img.width - 1, (x + 0.5) / r - 0.5);
      const x0 = Math.max(0, Math.floor(sx)), x1 = Math.min(img.width - 1, x0 + 1), fx = Math.max(0, sx - x0);
      const i00 = (y0 * img.width + x0) * 4, i01 = (y0 * img.width + x1) * 4;
      const i10 = (y1 * img.width + x0) * 4, i11 = (y1 * img.width + x1) * 4;
      const o = (y + top) * imgsz + (x + left);
      for (let c = 0; c < 3; c++) {
        const v = (img.data[i00 + c] * (1 - fx) + img.data[i01 + c] * fx) * (1 - fy) +
                  (img.data[i10 + c] * (1 - fx) + img.data[i11 + c] * fx) * fy;
        t[c * area + o] = v / 255;
      }
    }
  }
  return { tensor: t, r, left, top };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

// ---------- pós-processamento: saída [1, 4+nc, N] (YOLO11/YOLO26) ou [1, N, 6] (end-to-end) ----------
export function posprocessar(saida, dims, classes, prep, img, conf = 0.25, iouLim = 0.45) {
  const nc = classes.length;
  let cands = [];
  const desfaz = (x, y) => [(x - prep.left) / prep.r / img.width, (y - prep.top) / prep.r / img.height];
  if (dims.length === 3 && dims[1] === 4 + nc) {
    const n = dims[2];
    for (let i = 0; i < n; i++) {
      let best = 0, cls = -1;
      for (let c = 0; c < nc; c++) {
        const s = saida[(4 + c) * n + i];
        if (s > best) { best = s; cls = c; }
      }
      if (best < conf) continue;
      const cx = saida[i], cy = saida[n + i], w = saida[2 * n + i], h = saida[3 * n + i];
      const [x1, y1] = desfaz(cx - w / 2, cy - h / 2), [x2, y2] = desfaz(cx + w / 2, cy + h / 2);
      cands.push({ classe: classes[cls], conf: best, x1, y1, x2, y2 });
    }
    cands.sort((a, b) => b.conf - a.conf);
    const finais = [];
    for (const d of cands) {
      if (finais.every(f => f.classe !== d.classe || iou(f, d) < iouLim)) finais.push(d);
      if (finais.length >= 100) break;
    }
    cands = finais;
  } else if (dims.length === 3 && dims[2] === 6) {
    for (let i = 0; i < dims[1]; i++) {
      const o = i * 6, s = saida[o + 4];
      if (s < conf) continue;
      const [x1, y1] = desfaz(saida[o], saida[o + 1]), [x2, y2] = desfaz(saida[o + 2], saida[o + 3]);
      cands.push({ classe: classes[Math.round(saida[o + 5])], conf: s, x1, y1, x2, y2 });
    }
  } else {
    throw new Error("Formato de saída do modelo não reconhecido: " + dims.join("x"));
  }
  const lim = v => Math.min(1, Math.max(0, v));
  return cands.map(d => ({ ...d, x1: lim(d.x1), y1: lim(d.y1), x2: lim(d.x2), y2: lim(d.y2) }));
}

// ---------- regra de risco (igual a src/risco.py) ----------
export function coberturaAgua(dets) {
  const G = REGRAS.GRADE, marc = new Set();
  for (const d of dets) {
    if (d.classe !== "area_alagada" || d.conf < REGRAS.CONF_MIN) continue;
    const c1 = Math.floor(d.x1 * G), c2 = Math.min(G, Math.floor(d.x2 * G + 0.999));
    const l1 = Math.floor(d.y1 * G), l2 = Math.min(G, Math.floor(d.y2 * G + 0.999));
    for (let l = l1; l < l2; l++) for (let c = c1; c < c2; c++) marc.add(l * G + c);
  }
  return marc.size / (G * G);
}

export function nivelDaImagem(dets) {
  const val = dets.filter(d => d.conf >= REGRAS.CONF_MIN);
  const cob = coberturaAgua(val);
  const cls = new Set(val.map(d => d.classe));
  const pct = Math.round(cob * 100);
  if (cls.has("pessoa_na_agua")) return { nivel: 3, cobertura: cob, motivo: "pessoa detectada na água" };
  if (cls.has("veiculo_ilhado")) return { nivel: 2, cobertura: cob, motivo: "veículo ilhado detectado" };
  if (cob >= REGRAS.COBERTURA_ALTO) return { nivel: 2, cobertura: cob, motivo: `água cobre ${pct}% da imagem` };
  if (cob >= REGRAS.COBERTURA_ATENCAO || cls.has("area_alagada"))
    return { nivel: 1, cobertura: cob, motivo: `água cobre ${pct}% da imagem` };
  return { nivel: 0, cobertura: cob, motivo: "nenhuma detecção acima do limite de confiança" };
}

export class SuavizadorRisco {
  constructor(janela = 15, minQuadros = 6) { this.janela = janela; this.min = minQuadros; this.h = []; }
  atualizar(n) {
    this.h.push(n);
    if (this.h.length > this.janela) this.h.shift();
    for (const k of [3, 2, 1]) if (this.h.filter(x => x >= k).length >= this.min) return k;
    return 0;
  }
}

// ---------- execução ----------
export async function detectar(ort, sessao, meta, img, conf = 0.25) {
  const prep = preprocessar(img, meta.imgsz);
  const entrada = new ort.Tensor("float32", prep.tensor, [1, 3, meta.imgsz, meta.imgsz]);
  const out = await sessao.run({ [sessao.inputNames[0]]: entrada });
  const t = out[sessao.outputNames[0]];
  return posprocessar(t.data, t.dims, meta.classes, prep, img, conf);
}
