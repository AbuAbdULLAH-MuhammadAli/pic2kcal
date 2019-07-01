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
    return list(outs)
