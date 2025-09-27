# Fuzzy Automatic Game Settings (Mamdani)
import numpy as np

def trimf(x, abc):
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    idx = (a < x) & (x <= b)
    y[idx] = (x[idx] - a) / (b - a) if b != a else 0.0
    idx = (b < x) & (x < c)
    y[idx] = (c - x[idx]) / (c - b) if c != b else 0.0
    y[x == b] = 1.0
    return np.clip(y, 0, 1)

def trapmf(x, abcd):
    a, b, c, d = abcd
    y = np.zeros_like(x, dtype=float)
    idx = (a < x) & (x <= b)
    y[idx] = (x[idx] - a) / (b - a) if b != a else 0.0
    idx = (b < x) & (x <= c)
    y[idx] = 1.0
    idx = (c < x) & (x < d)
    y[idx] = (d - x[idx]) / (d - c) if d != c else 0.0
    y[(x == b) | (x == c)] = 1.0
    return np.clip(y, 0, 1)

def defuzz_centroid(x, mu):
    area = np.trapz(mu, x)
    if area == 0:
        return 0.5 * (x[0] + x[-1])
    return float(np.trapz(x * mu, x) / area)

class FuzzySet:
    def __init__(self, name, mf):
        self.name = name
        self.mf = mf

class FuzzyVar:
    def __init__(self, name, universe, sets):
        self.name = name
        self.universe = universe
        self.sets = sets
    def mu(self, set_name, xval):
        xs = self.universe
        mus = self.sets[set_name].mf(xs)
        return float(np.interp(xval, xs, mus))

class Rule:
    def __init__(self, antecedent, consequents):
        self.antecedent = antecedent
        self.consequents = consequents

def f_and(*vals): return float(np.min(vals))
def f_or(*vals):  return float(np.max(vals))
def f_not(val):   return 1.0 - float(val)

# Build variables
u_gpu   = np.linspace(0, 100, 1001)
u_net   = np.linspace(0, 200, 1001)
u_temp  = np.linspace(30, 100, 1001)
u_batt  = np.linspace(0, 100, 1001)

u_q     = np.linspace(0, 100, 1001)
u_res   = np.linspace(50, 100, 1001)
u_sh    = np.linspace(0, 100, 1001)
u_aa    = np.linspace(0, 100, 1001)

gpu_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (0, 0, 25, 45))),
    "mid":   FuzzySet("mid",   lambda x: trimf(x, (35, 55, 75))),
    "high":  FuzzySet("high",  lambda x: trapmf(x, (65, 80, 100, 100)))
}
net_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (0, 0, 20, 40))),
    "mid":   FuzzySet("mid",   lambda x: trimf(x, (30, 70, 110))),
    "high":  FuzzySet("high",  lambda x: trapmf(x, (90, 120, 200, 200)))
}
temp_sets = {
    "cool":  FuzzySet("cool",  lambda x: trapmf(x, (30, 30, 55, 65))),
    "warm":  FuzzySet("warm",  lambda x: trimf(x, (60, 70, 80))),
    "hot":   FuzzySet("hot",   lambda x: trapmf(x, (75, 85, 100, 100)))
}
batt_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (0, 0, 15, 30))),
    "mid":   FuzzySet("mid",   lambda x: trimf(x, (25, 45, 65))),
    "high":  FuzzySet("high",  lambda x: trapmf(x, (60, 75, 100, 100)))
}

quality_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (0, 0, 20, 35))),
    "med":   FuzzySet("med",   lambda x: trimf(x, (30, 50, 70))),
    "high":  FuzzySet("high",  lambda x: trimf(x, (60, 80, 90))),
    "ultra": FuzzySet("ultra", lambda x: trapmf(x, (85, 92, 100, 100)))
}
resscale_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (50, 50, 60, 75))),
    "med":   FuzzySet("med",   lambda x: trimf(x, (70, 80, 90))),
    "high":  FuzzySet("high",  lambda x: trapmf(x, (88, 94, 100, 100)))
}
shadows_sets = {
    "off":   FuzzySet("off",   lambda x: trapmf(x, (0, 0, 10, 25))),
    "low":   FuzzySet("low",   lambda x: trimf(x, (20, 35, 50))),
    "med":   FuzzySet("med",   lambda x: trimf(x, (45, 60, 75))),
    "high":  FuzzySet("high",  lambda x: trapmf(x, (70, 85, 100, 100)))
}
aa_sets = {
    "off":   FuzzySet("off",   lambda x: trapmf(x, (0, 0, 10, 20))),
    "low":   FuzzySet("low",   lambda x: trimf(x, (15, 30, 45))),
    "med":   FuzzySet("med",   lambda x: trimf(x, (40, 60, 75))),
    "high":  FuzzySet("high",  lambda x: trapmf(x, (70, 85, 100, 100)))
}

inputs = {
    "gpu":  FuzzyVar("gpu",  u_gpu,  gpu_sets),
    "net":  FuzzyVar("net",  u_net,  net_sets),
    "temp": FuzzyVar("temp", u_temp, temp_sets),
    "batt": FuzzyVar("batt", u_batt, batt_sets),
}
outputs = {
    "quality":  FuzzyVar("quality",  u_q,   quality_sets),
    "resscale": FuzzyVar("resscale", u_res, resscale_sets),
    "shadows":  FuzzyVar("shadows",  u_sh,  shadows_sets),
    "aa":       FuzzyVar("aa",       u_aa,  aa_sets),
}

def MU(var, set_name, x):
    return inputs[var].mu(set_name, x[var])

class MamdaniSystem:
    def __init__(self, inputs, outputs, rules):
        self.inputs = inputs
        self.outputs = outputs
        self.rules = rules
    def infer(self, invals):
        agg = {name: np.zeros_like(v.universe, dtype=float) for name, v in self.outputs.items()}
        for rule in self.rules:
            deg = float(rule.antecedent(invals))
            if deg <= 0: continue
            for out_name, set_name, weight in rule.consequents:
                out_var = self.outputs[out_name]
                mu_set = out_var.sets[set_name].mf(out_var.universe)
                clipped = np.minimum(mu_set, deg * weight)
                agg[out_name] = np.maximum(agg[out_name], clipped)
        crisp = {name: defuzz_centroid(v.universe, mu) for name, (v, mu) in zip(self.outputs.keys(), zip(self.outputs.values(), agg.values()))}
        return crisp

rules = []
rules.append(Rule(lambda x: f_and(MU("gpu","high",x), MU("net","low",x), MU("temp","cool",x), MU("batt","high",x)),
                 [("quality","ultra",1.0), ("resscale","high",1.0), ("shadows","high",1.0), ("aa","high",1.0)]))
rules.append(Rule(lambda x: f_and(MU("gpu","high",x), MU("net","mid",x), MU("temp","cool",x)),
                 [("quality","high",1.0), ("resscale","high",1.0), ("shadows","med",1.0), ("aa","med",1.0)]))
rules.append(Rule(lambda x: f_and(MU("gpu","mid",x), MU("net","low",x), MU("temp","cool",x)),
                 [("quality","med",1.0), ("resscale","med",1.0), ("shadows","med",1.0), ("aa","med",1.0)]))
rules.append(Rule(lambda x: f_and(MU("gpu","mid",x), MU("net","mid",x)),
                 [("quality","med",1.0), ("resscale","med",1.0), ("shadows","low",1.0), ("aa","low",1.0)]))
rules.append(Rule(lambda x: f_or(MU("gpu","low",x), MU("net","high",x)),
                 [("quality","low",1.0), ("resscale","low",1.0), ("shadows","off",1.0), ("aa","off",1.0)]))
rules.append(Rule(lambda x: MU("temp","hot",x),
                 [("quality","med",0.7), ("resscale","med",0.7), ("shadows","low",1.0), ("aa","low",1.0)]))
rules.append(Rule(lambda x: MU("temp","warm",x),
                 [("aa","low",0.7), ("shadows","low",0.7)]))
rules.append(Rule(lambda x: MU("batt","low",x),
                 [("quality","low",0.9), ("resscale","low",1.0), ("shadows","off",1.0), ("aa","off",1.0)]))
rules.append(Rule(lambda x: MU("batt","high",x),
                 [("resscale","high",0.4)]))
rules.append(Rule(lambda x: MU("net","low",x),
                 [("aa","med",0.5)]))
rules.append(Rule(lambda x: f_and(MU("gpu","high",x), MU("batt","low",x)),
                 [("aa","low",1.0), ("shadows","low",1.0)]))
rules.append(Rule(lambda x: f_and(MU("gpu","mid",x), MU("net","high",x)),
                 [("quality","low",1.0), ("resscale","low",1.0)]))

system = MamdaniSystem(inputs, outputs, rules)

def infer_settings(gpu, net, temp, batt):
    out = system.infer({"gpu": gpu, "net": net, "temp": temp, "batt": batt})
    return out

if __name__ == "__main__":
    ex = infer_settings(70, 40, 65, 80)
    print(ex)
