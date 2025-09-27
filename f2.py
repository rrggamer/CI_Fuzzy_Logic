import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

# -------- Membership helpers --------
def trimf(x, abc):
    a,b,c = abc
    y = np.zeros_like(x, dtype=float)
    idx = (a < x) & (x <= b); y[idx] = (x[idx]-a)/(b-a) if b!=a else 0.0
    idx = (b < x) & (x < c);  y[idx] = (c-x[idx])/(c-b) if c!=b else 0.0
    y[x==b] = 1.0
    return np.clip(y, 0, 1)

def trapmf(x, abcd):
    a,b,c,d = abcd
    y = np.zeros_like(x, dtype=float)
    idx = (a < x) & (x <= b); y[idx] = (x[idx]-a)/(b-a) if b!=a else 0.0
    idx = (b < x) & (x <= c); y[idx] = 1.0
    idx = (c < x) & (x < d);  y[idx] = (d-x[idx])/(d-c) if d!=c else 0.0
    y[(x==b)|(x==c)] = 1.0
    return np.clip(y, 0, 1)

def defuzz_centroid(x, mu):
    area = np.trapz(mu, x)
    if area == 0: return 0.5*(x[0]+x[-1])
    return float(np.trapz(x*mu, x)/area)

# -------- Fuzzy primitives --------
@dataclass
class FuzzySet:
    name: str
    mf: Callable[[np.ndarray], np.ndarray]

@dataclass
class FuzzyVar:
    name: str
    universe: np.ndarray
    sets: Dict[str, FuzzySet]
    def mu(self, set_name: str, xval: float) -> float:
        xs = self.universe; mus = self.sets[set_name].mf(xs)
        return float(np.interp(xval, xs, mus))

def f_and(*vals): return float(np.min(vals))
def f_or(*vals):  return float(np.max(vals))

@dataclass
class Rule:
    antecedent: Callable[[Dict[str,float]], float]
    consequents: List[Tuple[str, str, float]]  # (out_var, set_name, weight)

class MamdaniSystem:
    def __init__(self, inputs: Dict[str,FuzzyVar], outputs: Dict[str,FuzzyVar], rules: List[Rule]):
        self.inputs, self.outputs, self.rules = inputs, outputs, rules
    def infer(self, x: Dict[str,float]) -> Dict[str,float]:
        agg = {name: np.zeros_like(var.universe) for name, var in self.outputs.items()}
        for rule in self.rules:
            w = float(rule.antecedent(x))
            if w <= 0: continue
            for out_name, set_name, weight in rule.consequents:
                var = self.outputs[out_name]
                mu  = var.sets[set_name].mf(var.universe)
                agg[out_name] = np.maximum(agg[out_name], np.minimum(mu, w*weight))
        return {name: defuzz_centroid(var.universe, agg[name]) for name, var in self.outputs.items()}

# -------- Build variables --------
u  = np.linspace(0,100,1001)
uR = np.linspace(50,100,1001)

def mk_perf_sets():
    return {
        "low":  FuzzySet("low",  lambda x: trapmf(x, (0,0,25,45))),
        "mid":  FuzzySet("mid",  lambda x: trimf(x, (35,55,75))),
        "high": FuzzySet("high", lambda x: trapmf(x, (65,80,100,100))),
    }

inputs = {
    "cpu": FuzzyVar("cpu", u, mk_perf_sets()),
    "gpu": FuzzyVar("gpu", u, mk_perf_sets()),
    "ram": FuzzyVar("ram", u, mk_perf_sets()),
}

quality_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (0,0,25,40))),
    "med":   FuzzySet("med",   lambda x: trimf(x, (35,55,70))),
    "high":  FuzzySet("high",  lambda x: trimf(x, (65,80,90))),
    "ultra": FuzzySet("ultra", lambda x: trapmf(x, (85,92,100,100))),
}
res_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (50,50,60,75))),
    "med":   FuzzySet("med",   lambda x: trimf(x, (70,80,90))),
    "high":  FuzzySet("high",  lambda x: trapmf(x, (88,94,100,100))),
}
tex_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (0,0,30,50))),
    "med":   FuzzySet("med",   lambda x: trimf(x, (45,60,75))),
    "high":  FuzzySet("high",  lambda x: trapmf(x, (70,85,100,100))),
}

outputs = {
    "quality": FuzzyVar("quality", u,  quality_sets),
    "res":     FuzzyVar("res",     uR, res_sets),
    "tex":     FuzzyVar("tex",     u,  tex_sets),
}

def MU(v, s, x): return inputs[v].mu(s, x[v])

# -------- Human-readable rules --------
RULES_HUMAN = [
    "R1: IF CPU is HIGH AND GPU is HIGH AND RAM is HIGH THEN Quality is ULTRA, Resolution is HIGH, Texture is HIGH",
    "R2: IF GPU is HIGH AND CPU is MID AND RAM is MID THEN Quality is HIGH, Resolution is HIGH, Texture is MED",
    "R3: IF CPU is MID AND GPU is MID AND RAM is MID THEN Quality is MED, Resolution is MED, Texture is MED",
    "R4: IF GPU is LOW OR CPU is LOW THEN Quality is LOW, Resolution is LOW",
    "R5: IF RAM is LOW THEN Texture is LOW",
    "R6: IF GPU is HIGH AND RAM is MID THEN Quality is HIGH, Resolution is HIGH, Texture is MED (weighted)",
    "R7: IF CPU is MID AND GPU is HIGH AND RAM is HIGH THEN Quality is HIGH, Resolution is HIGH, Texture is HIGH",
    "R8: IF CPU is HIGH AND GPU is MID AND RAM is HIGH THEN Quality is HIGH (slightly), Resolution is MED, Texture is HIGH",
    "R9: IF GPU is MID AND RAM is LOW THEN Texture is LOW; Resolution tends to LOW",
    "R10: IF CPU is LOW AND GPU is HIGH THEN Quality is MED, Resolution is MED",
]

# Machine rules (with weights)
RULES = [
    Rule(lambda x: f_and(MU('cpu','high',x), MU('gpu','high',x), MU('ram','high',x)),
         [('quality','ultra',1.0), ('res','high',1.0), ('tex','high',1.0)]),
    Rule(lambda x: f_and(MU('gpu','high',x), MU('cpu','mid',x), MU('ram','mid',x)),
         [('quality','high',1.0), ('res','high',1.0), ('tex','med',1.0)]),
    Rule(lambda x: f_and(MU('cpu','mid',x), MU('gpu','mid',x), MU('ram','mid',x)),
         [('quality','med',1.0), ('res','med',1.0), ('tex','med',1.0)]),
    Rule(lambda x: f_or(MU('gpu','low',x), MU('cpu','low',x)),
         [('quality','low',1.0), ('res','low',1.0)]),
    Rule(lambda x: MU('ram','low',x),
         [('tex','low',1.0)]),
    Rule(lambda x: f_and(MU('gpu','high',x), MU('ram','mid',x)),
         [('quality','high',0.9), ('res','high',0.9), ('tex','med',1.0)]),
    Rule(lambda x: f_and(MU('cpu','mid',x), MU('gpu','high',x), MU('ram','high',x)),
         [('quality','high',1.0), ('res','high',1.0), ('tex','high',1.0)]),
    Rule(lambda x: f_and(MU('cpu','high',x), MU('gpu','mid',x), MU('ram','high',x)),
         [('quality','high',0.8), ('res','med',1.0), ('tex','high',1.0)]),
    Rule(lambda x: f_and(MU('gpu','mid',x), MU('ram','low',x)),
         [('tex','low',1.0), ('res','low',0.8)]),
    Rule(lambda x: f_and(MU('cpu','low',x), MU('gpu','high',x)),
         [('quality','med',1.0), ('res','med',1.0)]),
]

system = MamdaniSystem(inputs, outputs, RULES)

# -------- Numeric → Label mapping --------
def quality_label(q: float) -> str:
    if q < 37.5: return "Low"
    if q < 67.5: return "Medium"
    if q < 87.5: return "High"
    return "Ultra"

def resolution_label(res: float) -> str:
    if res < 72.5: return "Low"
    if res < 89.0: return "Medium"
    return "High"

def texture_label(tex: float) -> str:
    if tex < 47.5: return "Low"
    if tex < 72.5: return "Medium"
    return "High"

class HysteresisMapper:
    def __init__(self, levels, rise, fall, initial=None):
        self.levels, self.rise, self.fall = levels, rise, fall
        self.idx = 0 if initial is None else levels.index(initial)

    def update(self, x: float) -> str:
        while self.idx < len(self.rise) and x >= self.rise[self.idx]:
            self.idx += 1
        while self.idx > 0 and x < self.fall[self.idx-1]:
            self.idx -= 1
        return self.levels[self.idx]

quality_mapper = HysteresisMapper(["Low","Medium","High","Ultra"], [40.0,70.0,90.0], [35.0,65.0,85.0])
res_mapper     = HysteresisMapper(["Low","Medium","High"], [75.0,90.0], [70.0,88.0])
tex_mapper     = HysteresisMapper(["Low","Medium","High"], [50.0,75.0], [45.0,70.0])

# -------- Public APIs --------
def infer(cpu: float, gpu: float, ram: float) -> Dict[str,float]:
    return system.infer({"cpu": cpu, "gpu": gpu, "ram": ram})

def infer_with_labels(cpu: float, gpu: float, ram: float, use_hysteresis: bool=False) -> Dict[str,object]:
    out = infer(cpu, gpu, ram)
    if use_hysteresis:
        qlbl = quality_mapper.update(out["quality"])
        rlbl = res_mapper.update(out["res"])
        tlbl = tex_mapper.update(out["tex"])
    else:
        qlbl = quality_label(out["quality"])
        rlbl = resolution_label(out["res"])
        tlbl = texture_label(out["tex"])
    return {
        "cpu": cpu, "gpu": gpu, "ram": ram,
        "quality_idx": round(out["quality"],2), "quality_lbl": qlbl,
        "res_scale_%": round(out["res"],1),    "res_label": rlbl,
        "texture_idx": round(out["tex"],1),    "texture_lbl": tlbl,
    }


if __name__ == "__main__":
    print(infer_with_labels(90, 40, 95))

