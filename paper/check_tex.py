"""Static sanity checks for main.tex (no LaTeX installation needed).  python paper/check_tex.py"""
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
s = (HERE / "main.tex").read_text(encoding="utf-8")
bib = (HERE / "references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{(\w+),", bib))
cites = {k.strip() for g in re.findall(r"\\cite[pt]?\{([^}]*)\}", s) for k in g.split(",")}
print("missing cite keys:", cites - keys or "none", "| unused bib keys:", keys - cites or "none")
INPUT = r"\\input(?:table)?\{([^}]*?)(?:\.tex)?\}"   # \input{x} 또는 \inputtable{x.tex}
for f in re.findall(INPUT, s):
    if not (HERE / (f + ".tex")).exists():
        print("MISSING input:", f)
for f in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", s):
    if not (HERE / f).exists():
        print("MISSING figure:", f)
body = re.sub(r"(?<!\\)%.*", "", s)
print("brace balance:", body.count("{") - body.count("}"))
print("begin/end:", len(re.findall(r"\\begin\{", body)), len(re.findall(r"\\end\{", body)))
labels = set(re.findall(r"\\label\{([^}]*)\}", s))
refs = set(re.findall(r"\\ref\{([^}]*)\}", s))
print("undefined refs:", refs - labels or "none")

# column counts: tabular spec vs rows of each \input-ed table
for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", s, re.S):
    ncol = len(re.sub(r"[^lcr]", "", m.group(1)))
    for f in re.findall(INPUT, m.group(2)):
        for line in (HERE / (f + ".tex")).read_text(encoding="utf-8").splitlines():
            if line.strip().endswith("\\\\") and "multicolumn" not in line:
                n = line.count("&") + 1
                if n != ncol:
                    print(f"COLUMN MISMATCH {f}: tabular {ncol}, row {n}: {line[:60]}")
print("done")
