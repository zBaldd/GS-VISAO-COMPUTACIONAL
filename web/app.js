import { ROTULOS, NIVEIS, SuavizadorRisco, detectar, nivelDaImagem } from "./detector.js";

const ort = window.ort;
const $ = id => document.getElementById(id);
const CORES = { area_alagada: "#1e8ce6", veiculo_ilhado: "#ff9800", pessoa_na_agua: "#e53935" };
const LARGURA_ANALISE = 960; // reduz fotos grandes antes de analisar (mais rápido no celular)

let sessao = null, meta = null, camera = null, ultimo = null;
const suav = new SuavizadorRisco(15, 6);

function status(msg) { $("status").textContent = msg; }

async function carregarModelo() {
  try {
    if (!ort) throw new Error("onnxruntime-web não carregou (sem internet?)");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
    meta = await (await fetch("model/meta.json", { cache: "no-cache" })).json();
    sessao = await ort.InferenceSession.create("model/alagamento.onnx", { executionProviders: ["wasm"] });
    const m = meta.metricas_teste;
    $("info-modelo").textContent = `${meta.origem} · ${meta.imgsz}px · ${meta.classes.length} classes` +
      (m ? ` · mAP50 no teste ${(m.mAP50 * 100).toFixed(1)}%` : "");
    status("Modelo pronto. Envie uma foto ou ligue a câmera.");
  } catch (e) {
    console.error(e);
    status("Não foi possível carregar a detecção automática. Você ainda pode registrar o relato escolhendo o nível manualmente.");
    $("nivel-final").disabled = false;
    $("enviar").disabled = false;
  }
}

function imagemParaDados(fonte, w, h) {
  const esc = Math.min(1, LARGURA_ANALISE / w);
  const c = document.createElement("canvas");
  c.width = Math.round(w * esc); c.height = Math.round(h * esc);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(fonte, 0, 0, c.width, c.height);
  return ctx.getImageData(0, 0, c.width, c.height);
}

function desenhar(fonte, w, h, dets) {
  const tela = $("tela");
  tela.width = w; tela.height = h; tela.hidden = false; $("vazio").hidden = true;
  const ctx = tela.getContext("2d");
  ctx.drawImage(fonte, 0, 0, w, h);
  const esp = Math.max(2, Math.round(w / 300));
  ctx.font = `600 ${Math.max(12, Math.round(w / 45))}px system-ui, sans-serif`;
  const ordem = [...dets].sort((a, b) => (a.classe === "area_alagada" ? -1 : 1) - (b.classe === "area_alagada" ? -1 : 1));
  for (const d of ordem) {
    const x = d.x1 * w, y = d.y1 * h, bw = (d.x2 - d.x1) * w, bh = (d.y2 - d.y1) * h;
    ctx.strokeStyle = CORES[d.classe]; ctx.lineWidth = esp;
    if (d.classe === "area_alagada") { ctx.fillStyle = "rgba(30,140,230,.18)"; ctx.fillRect(x, y, bw, bh); }
    ctx.strokeRect(x, y, bw, bh);
    const txt = `${ROTULOS[d.classe]} ${Math.round(d.conf * 100)}%`;
    const tw = ctx.measureText(txt).width + 10, th = parseInt(ctx.font.match(/\d+px/)[0]) + 8;
    ctx.fillStyle = CORES[d.classe]; ctx.fillRect(x, Math.max(0, y - th), tw, th);
    ctx.fillStyle = "#fff"; ctx.fillText(txt, x + 5, Math.max(th - 6, y - 6));
  }
}

function mostrarRisco(nivel, r, dets) {
  const box = $("risco");
  box.className = `risco n${nivel}`;
  $("risco-t").textContent = NIVEIS[nivel].titulo;
  $("risco-o").textContent = NIVEIS[nivel].orientacao;
  $("motivo").textContent = `Motivo: ${r.motivo}. Água estimada em ${Math.round(r.cobertura * 100)}% da imagem.`;
  $("dets").innerHTML = dets.length
    ? dets.map(d => `<li><span>${ROTULOS[d.classe]}</span><span class="pill">${Math.round(d.conf * 100)}%</span></li>`).join("")
    : "<li><span>Nenhuma detecção</span></li>";
  $("emergencia").hidden = nivel !== 3;
  $("nivel-final").value = String(nivel);
  $("nivel-final").disabled = false;
  $("enviar").disabled = false;
  ultimo = { nivel, cobertura: r.cobertura, classes: [...new Set(dets.filter(d => d.conf >= 0.35).map(d => d.classe))] };
}

async function analisarFoto(arquivo) {
  if (!arquivo) return;
  pararCamera();
  const bmp = await createImageBitmap(arquivo, { imageOrientation: "from-image" });
  const dados = imagemParaDados(bmp, bmp.width, bmp.height);
  if (!sessao) { desenhar(bmp, dados.width, dados.height, []); return; }
  status("Analisando a foto…");
  const t0 = performance.now();
  const dets = await detectar(ort, sessao, meta, dados, 0.25);
  const r = nivelDaImagem(dets);
  desenhar(bmp, dados.width, dados.height, dets);
  mostrarRisco(r.nivel, r, dets);
  status(`Análise feita no aparelho em ${Math.round(performance.now() - t0)} ms.`);
}

async function ligarCamera() {
  if (camera) { pararCamera(); return; }
  if (!sessao) { status("O modelo ainda não carregou."); return; }
  try {
    camera = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
  } catch (e) {
    status("Não foi possível abrir a câmera (permissão negada ou indisponível). Use a opção de foto.");
    return;
  }
  const v = $("video"); v.srcObject = camera; await v.play();
  $("btn-camera").textContent = "⏹ Parar câmera";
  let quadros = 0, t0 = performance.now();
  const loop = async () => {
    if (!camera) return;
    const dados = imagemParaDados(v, v.videoWidth, v.videoHeight);
    const dets = await detectar(ort, sessao, meta, dados, 0.25);
    const r = nivelDaImagem(dets);
    const nivel = suav.atualizar(r.nivel); // filtro temporal: evita alerta por 1 quadro errado
    desenhar(v, dados.width, dados.height, dets);
    mostrarRisco(nivel, r, dets);
    quadros++;
    const s = (performance.now() - t0) / 1000;
    if (s > 1) { status(`Câmera ao vivo · ${(quadros / s).toFixed(1)} análises por segundo`); quadros = 0; t0 = performance.now(); }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function pararCamera() {
  if (!camera) return;
  camera.getTracks().forEach(t => t.stop());
  camera = null;
  $("btn-camera").textContent = "🎥 Câmera ao vivo";
  status("Câmera desligada.");
}

// ---- relatos (guardados só neste navegador; no produto, vão para a API/RPA) ----
function lerRelatos() { try { return JSON.parse(localStorage.getItem("relatos_visao") || "[]"); } catch { return []; } }
function salvarRelatos(l) { try { localStorage.setItem("relatos_visao", JSON.stringify(l)); } catch { /* modo privado */ } }
function atualizarLista() { const l = lerRelatos(); $("qtd").textContent = l.length; $("json").textContent = JSON.stringify(l, null, 2); }

$("enviar").addEventListener("click", () => {
  const final = Number($("nivel-final").value);
  const relato = {
    data_hora: new Date().toISOString(),
    municipio: $("municipio").value,
    nivel_agua: final,
    nivel_texto: NIVEIS[final].titulo,
    nivel_sugerido_ia: ultimo ? ultimo.nivel : null,
    corrigido_pelo_cidadao: ultimo ? ultimo.nivel !== final : null,
    cobertura_agua: ultimo ? Math.round(ultimo.cobertura * 1000) / 1000 : null,
    classes_detectadas: ultimo ? ultimo.classes : [],
    origem: ultimo ? "foto_visao_computacional" : "manual",
    modelo: meta ? meta.origem : null,
  };
  const l = lerRelatos(); l.push(relato); salvarRelatos(l); atualizarLista();
  status(`Relato registrado: ${relato.nivel_texto} em ${relato.municipio}. Obrigado!`);
});

$("baixar").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(lerRelatos(), null, 2)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "relatos_visao.json" });
  a.click(); URL.revokeObjectURL(a.href);
});

$("arquivo").addEventListener("change", e => analisarFoto(e.target.files[0]).catch(err => { console.error(err); status("Não consegui analisar essa imagem."); }));
$("btn-foto").addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("arquivo").click(); } });
$("btn-camera").addEventListener("click", ligarCamera);

atualizarLista();
carregarModelo();
