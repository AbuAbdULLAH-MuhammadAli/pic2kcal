import re


def normalize_out_ingredient(ing: str):
    regs = []  # [r", frisch$", r", roh$", r", vom Huhn$", r", grün$"]

    ing = ing.strip()
    outs = {ing}
    while True:
        # Zucchini, grün, frisch -> Zucchini UND Zucchini, grün UND Zucchini, grün, frisch
        ing2 = re.sub(r",[^,]+$", "", ing)
        if ing == ing2:
            break
        ing = ing2
        outs.add(ing)

    additions = {
        "Ei": ["Eier"],
        "Salz": ["Salz und Pfeffer"],
        "Paprika": ["Paprikaschote"],
        "Knoblauch": ["Knoblauchzehe"],
        "Karotten / Möhren": ["Karotten", "Karotte", "Möhren", "Möhre"],
        "Lorbeerblatt": ["Lorbeer"],
        "Apfel": ["Äpfel"]
    }
    for reg in regs:
        ing = re.sub(reg, "", ing)
        outs.add(ing)
    if ing in additions:
        outs.update(additions[ing])
    return list(outs)
