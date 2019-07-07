import re
def normalize_out_ingredient(ing: str):
    regs = [r", frisch$", r", roh$", r", vom Huhn$", r", grün$"]
    ing = ing.strip()
    outs = {ing}
    for reg in regs:
        ing = re.sub(reg, "", ing)
        outs.add(ing)
    if ing == "Ei":
        outs.add("Eier")
    if ing == "Salz":
        outs.add("Salz und Pfeffer")
    if ing == "Karotten / Möhren":
        outs.add("Karotten")
        outs.add("Möhren")
    return list(outs)
