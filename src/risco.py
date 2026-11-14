"""Transforma as detecções do YOLO em um nível de risco para o cidadão.

O modelo só diz "o que" e "onde". A decisão de risco é uma regra explícita,
escrita aqui para que qualquer pessoa consiga auditar por que um alerta saiu
(princípio de IA responsável do briefing). Os mesmos limites são usados no
navegador (web/app.js) — se mudar aqui, mude lá.
"""
from collections import deque
from dataclasses import dataclass, field

NIVEIS = {
    0: ("SEM ALAGAMENTO", "Nenhum sinal de alagamento na imagem."),
    1: ("ATENCAO", "Agua na via. Reduza a velocidade e evite poças fundas."),
    2: ("ALTO", "Via possivelmente intransitavel. Procure rota alternativa."),
    3: ("CRITICO", "Pessoa em risco na agua. Ligue 193 (Bombeiros) ou 199 (Defesa Civil)."),
}

# limites (documentados no relatório, seção Metodologia)
CONF_MIN = 0.35            # confiança mínima para considerar uma detecção
COBERTURA_ATENCAO = 0.05   # >= 5% da imagem com água -> ATENÇÃO
COBERTURA_ALTO = 0.35      # >= 35% da imagem com água -> ALTO
GRADE = 64                 # resolução da grade usada para medir a cobertura


@dataclass
class Deteccao:
    classe: str
    conf: float
    x1: float  # coordenadas normalizadas 0..1
    y1: float
    x2: float
    y2: float


def cobertura_agua(deteccoes, grade: int = GRADE) -> float:
    """Fração da imagem coberta pela UNIÃO das caixas de area_alagada (0..1).

    Usa uma grade grade x grade para não contar duas vezes caixas sobrepostas.
    """
    marcadas = set()
    for d in deteccoes:
        if d.classe != "area_alagada" or d.conf < CONF_MIN:
            continue
        c1, c2 = int(max(0, d.x1) * grade), int(min(1, d.x2) * grade + 0.999)
        l1, l2 = int(max(0, d.y1) * grade), int(min(1, d.y2) * grade + 0.999)
        for l in range(l1, min(l2, grade)):
            for c in range(c1, min(c2, grade)):
                marcadas.add((l, c))
    return len(marcadas) / (grade * grade)


def nivel_da_imagem(deteccoes):
    """Regra de decisão para UMA imagem. Devolve (nivel, cobertura, motivo)."""
    validas = [d for d in deteccoes if d.conf >= CONF_MIN]
    cob = cobertura_agua(validas)
    classes = {d.classe for d in validas}

    if "pessoa_na_agua" in classes:
        return 3, cob, "pessoa detectada na agua"
    if "veiculo_ilhado" in classes:
        return 2, cob, "veiculo ilhado detectado"
    if cob >= COBERTURA_ALTO:
        return 2, cob, f"agua cobre {cob:.0%} da imagem"
    if cob >= COBERTURA_ATENCAO or "area_alagada" in classes:
        return 1, cob, f"agua cobre {cob:.0%} da imagem"
    return 0, cob, "nenhuma deteccao acima do limite"


@dataclass
class SuavizadorRisco:
    """Filtro temporal para vídeo em tempo real.

    Um quadro isolado com falso positivo não dispara alerta: o nível N só é
    exibido se pelo menos `min_quadros` dos últimos `janela` quadros tiverem
    nível >= N. Isso dá estabilidade (histerese) sem atrasar demais o alerta.
    """
    janela: int = 15
    min_quadros: int = 6
    _hist: deque = field(default_factory=deque)

    def atualizar(self, nivel: int) -> int:
        self._hist.append(nivel)
        while len(self._hist) > self.janela:
            self._hist.popleft()
        for n in (3, 2, 1):
            if sum(1 for x in self._hist if x >= n) >= self.min_quadros:
                return n
        return 0
