"""Classes do detector e mapa de sinônimos usado para unir datasets públicos.

Os datasets abertos de enchente usam nomes diferentes para a mesma coisa
("flooded_area", "flood", "water"...). Aqui definimos as 3 classes do
Alerta Chuva SP e de quais nomes originais cada uma vem.
"""

CLASSES = ["area_alagada", "veiculo_ilhado", "pessoa_na_agua"]

ROTULOS_PT = {
    "area_alagada": "Rua alagada",
    "veiculo_ilhado": "Veiculo ilhado",
    "pessoa_na_agua": "Pessoa na agua",
}

# nome original (minúsculo, sem espaços extras) -> classe do projeto
# nomes que não estão aqui são descartados no preparo do dataset
SINONIMOS = {
    # área alagada / via com água
    "flooded_area": "area_alagada",
    "flooded area": "area_alagada",
    "flooded_road": "area_alagada",
    "flooded road": "area_alagada",
    "flood": "area_alagada",
    "flooded": "area_alagada",
    "floodwater": "area_alagada",
    "flood_water": "area_alagada",
    "water": "area_alagada",
    "water_on_road": "area_alagada",
    "area_alagada": "area_alagada",
    # veículo ilhado
    "flooded_car": "veiculo_ilhado",
    "flooded car": "veiculo_ilhado",
    "drown car": "veiculo_ilhado",
    "drowning car": "veiculo_ilhado",
    "car_in_water": "veiculo_ilhado",
    "tier_merged_car": "veiculo_ilhado",
    "flooded_vehicle": "veiculo_ilhado",
    "veiculo_ilhado": "veiculo_ilhado",
    # pessoa na água
    "flooded_person": "pessoa_na_agua",
    "flooded person": "pessoa_na_agua",
    "human in flood": "pessoa_na_agua",
    "person_in_help": "pessoa_na_agua",
    "in water": "pessoa_na_agua",
    "person_in_water": "pessoa_na_agua",
    "pessoa_na_agua": "pessoa_na_agua",
}


def classe_do_projeto(nome_original: str):
    """Devolve a classe do projeto para um nome original, ou None se for descartado."""
    chave = " ".join(str(nome_original).strip().lower().split())
    if chave in SINONIMOS:
        return SINONIMOS[chave]
    return SINONIMOS.get(chave.replace(" ", "_"))
