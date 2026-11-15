"""Testes da regra de risco e do mapa de classes. Rode com: python -m pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classes import classe_do_projeto  # noqa: E402
from risco import Deteccao, SuavizadorRisco, cobertura_agua, nivel_da_imagem  # noqa: E402


def d(classe, conf=0.9, x1=0.0, y1=0.0, x2=0.1, y2=0.1):
    return Deteccao(classe, conf, x1, y1, x2, y2)


def test_sem_deteccao_e_nivel_zero():
    assert nivel_da_imagem([])[0] == 0


def test_pessoa_na_agua_e_critico():
    assert nivel_da_imagem([d("pessoa_na_agua")])[0] == 3


def test_veiculo_ilhado_e_alto():
    assert nivel_da_imagem([d("veiculo_ilhado")])[0] == 2


def test_cobertura_define_atencao_e_alto():
    assert nivel_da_imagem([d("area_alagada", x2=0.3, y2=0.3)])[0] == 1   # 9%
    assert nivel_da_imagem([d("area_alagada", x2=0.7, y2=0.7)])[0] == 2   # 49%


def test_baixa_confianca_e_ignorada():
    assert nivel_da_imagem([d("pessoa_na_agua", conf=0.2)])[0] == 0


def test_caixas_sobrepostas_nao_contam_duas_vezes():
    a = d("area_alagada", x2=0.5, y2=0.5)
    assert abs(cobertura_agua([a, a]) - 0.25) < 0.01


def test_suavizador_ignora_quadro_isolado():
    s = SuavizadorRisco(janela=10, min_quadros=4)
    niveis = [s.atualizar(n) for n in [0, 0, 3, 0, 0, 0]]
    assert max(niveis) == 0
    niveis = [s.atualizar(2) for _ in range(5)]
    assert niveis[-1] == 2


def test_sinonimos():
    assert classe_do_projeto("Flooded_Car") == "veiculo_ilhado"
    assert classe_do_projeto("flooded road") == "area_alagada"
    assert classe_do_projeto("human in flood") == "pessoa_na_agua"
    assert classe_do_projeto("house") is None


def test_original_roboflow_agrupa_copias():
    from preparar_dataset import original_roboflow
    assert original_roboflow("rua_12_jpg.rf.ab12cd.jpg") == original_roboflow("rua_12_jpg.rf.ff0099.jpg")
    assert original_roboflow("foto.jpg") == "foto.jpg"


def test_resplit_estratifica_classes_raras():
    from pathlib import Path
    from preparar_dataset import refazer_split
    regs = []
    for i in range(80):
        presentes = {"pessoa_na_agua"} if i < 16 else {"area_alagada"}
        for k in range(3):  # 3 cópias aumentadas de cada foto
            regs.append({"img": Path(f"f{i}_jpg.rf.{k}.jpg"), "presentes": presentes, "grupo": f"d0:f{i}"})
    out = refazer_split(regs, [0.75, 0.125, 0.125], 42)
    pessoas_teste = sum(1 for r, s in out if s == "test" and "pessoa_na_agua" in r["presentes"])
    assert pessoas_teste == 2  # 12,5% de 16
    grupos = {}
    for r, s in out:
        assert grupos.setdefault(r["grupo"], s) == s  # nenhuma foto em dois splits
    assert all(sum(1 for r, s2 in out if r["grupo"] == g) == 1 for g, s in grupos.items() if s != "train")
