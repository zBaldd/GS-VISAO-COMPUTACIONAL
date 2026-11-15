| Split | Imagens | Sem rótulo (negativas) | area_alagada | veiculo_ilhado | pessoa_na_agua |
|---|---|---|---|---|---|
| train | 3944 | 267 | 1064 | 2503 | 5979 |
| valid | 531 | 27 | 125 | 409 | 930 |
| test | 530 | 27 | 112 | 356 | 865 |

Split: refeito por foto original, estratificado
Duplicadas removidas: 476
Imagens enviadas para pseudo-rotulagem de água: 3668

Origem das classes:
- **area_alagada** ← flood-detection-aicourse:flooded_road, flood-nfc4z:flood, flooding-object-detection-opiei:flooded_area
- **veiculo_ilhado** ← flood-cka5z:car, flood-detection-aicourse:flooded_car, flood-object-detection-2olqp:flooded_car, flooded-objects:flooded car, flooding-object-detection-opiei:flooded_car
- **pessoa_na_agua** ← flood-cka5z:person, flood-object-detection-2olqp:person, flood-object-detection-2olqp:person_in_help, flood-person-detection-uu8x0:person_in_water, flooded-objects:human in flood, flooding-object-detection-opiei:flooded_person

Classes descartadas: flood-cka5z:house, flood-detection-aicourse:flooded_electric_distribution_pole, flood-detection-aicourse:flooded_fallen_trees, flood-detection-aicourse:flooded_house, flood-object-detection-2olqp:flooded_building, flood-person-detection-uu8x0:person_normal, flood-person-detection-uu8x0:person_on_boat, flooded-objects:This dataset was exported via roboflow.com on November 5- 2024 at 9-19 AM GMT, flooding-object-detection-opiei:flooded_building
