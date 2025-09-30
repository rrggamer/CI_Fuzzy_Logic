import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple
import pandas as pd
import os

# ============================================
# โครงงาน: Fuzzy (Mamdani) เลือกการตั้งค่าเกมอัตโนมัติ
# อินพุต: CPU/GPU/RAM (0..100)
# เอาต์พุต: Graphics Quality (0..100), Resolution Scale (50..100), Texture (0..100)
# ขั้นตอน: fuzzification -> evaluate rules (AND=min, OR=max)
#          -> implication=min -> aggregation=max -> centroid defuzz
# ============================================

# -------- Membership helpers --------
def trimf(x, abc):
    """ Triangular MF: จุด (a,b,c) — ไต่ขึ้น a->b แล้วไต่ลง b->c """
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    # ไต่ขึ้น
    idx = (a < x) & (x <= b); y[idx] = (x[idx] - a) / (b - a) if b != a else 0.0
    # ไต่ลง
    idx = (b < x) & (x < c);  y[idx] = (c - x[idx]) / (c - b) if c != b else 0.0
    # จุดยอด (b) = 1
    y[x == b] = 1.0
    return np.clip(y, 0, 1)

def trapmf(x, abcd):
    """ Trapezoidal MF: จุด (a,b,c,d) — ไต่ขึ้น a->b, ราบ b..c, ไต่ลง c->d """
    a, b, c, d = abcd
    y = np.zeros_like(x, dtype=float)
    # ไต่ขึ้น
    idx = (a < x) & (x <= b); y[idx] = (x[idx] - a) / (b - a) if b != a else 0.0
    # ช่วงราบ
    idx = (b < x) & (x <= c); y[idx] = 1.0
    # ไต่ลง
    idx = (c < x) & (x < d);  y[idx] = (d - x[idx]) / (d - c) if d != c else 0.0
    y[(x == b) | (x == c)] = 1.0
    return np.clip(y, 0, 1)

def defuzz_centroid(x, mu):
    """ Centroid of area (CoA): ค่าศูนย์ถ่วงของพื้นที่ใต้โค้ง mu(x) """
    area = np.trapz(mu, x)
    if area == 0:
        # ถ้าไม่มีพื้นที่ (ไม่มีกฎลั่น) คืนค่ากลางโดเมนเพื่อกันหารศูนย์
        return 0.5 * (x[0] + x[-1])
    return float(np.trapz(x * mu, x) / area)


# -------- โครงสร้างพื้นฐานของระบบฟัซซี --------
class FuzzySet:
    """ 1 ชุดสังกัด (เช่น 'low', 'mid', 'high') พร้อมฟังก์ชัน membership """
    def __init__(self, name: str, mf: Callable[[np.ndarray], np.ndarray]):
        self.name = name
        self.mf = mf

class FuzzyVar:
    """ ตัวแปรฟัซซี 1 ตัว: มีโดเมน (universe) + เซตย่อย (FuzzySet) หลายชุด """
    def __init__(self, name: str, universe: np.ndarray, sets: Dict[str, FuzzySet]):
        self.name = name
        self.universe = universe
        self.sets = sets

    def mu(self, set_name: str, xval: float) -> float:
        """ อ่านค่า membership (0..1) ของ set_name ที่ตำแหน่ง xval """
        xs = self.universe
        mus = self.sets[set_name].mf(xs)
        return float(np.interp(xval, xs, mus))  # interpolate ให้ค่าเดียวจากโค้ง

class Rule:
    """ กฎฟัซซี: antecedent (degree 0..1) -> consequents [(out_var, set_name, weight)] """
    def __init__(self,
                 antecedent: Callable[[Dict[str, float]], float],
                 consequents: List[Tuple[str, str, float]]):
        self.antecedent = antecedent
        self.consequents = consequents

class MamdaniSystem:
    """ แกนหลักของ Mamdani: implication=min, aggregation=max, defuzz=centroid """
    def __init__(self, inputs: Dict[str, FuzzyVar], outputs: Dict[str, FuzzyVar], rules: List[Rule]):
        self.inputs, self.outputs, self.rules = inputs, outputs, rules

    def infer(self, x: Dict[str, float]) -> Dict[str, float]:
        # เริ่มกราฟสะสมของแต่ละเอาต์พุตเป็นศูนย์ (สำหรับ aggregation)
        agg = {name: np.zeros_like(var.universe, dtype=float) for name, var in self.outputs.items()}
        for rule in self.rules:
            # 1) ประเมินส่วน IF: AND=min, OR=max (นิยามใน lambda ของ RULES)
            w = float(rule.antecedent(x))
            if w <= 0:
                continue
            # 2) implication = min: clip ชุดสังกัดของเอาต์พุตด้วยระดับ w (คูณน้ำหนักได้)
            for out_name, set_name, weight in rule.consequents:
                var = self.outputs[out_name]
                mu_set = var.sets[set_name].mf(var.universe)
                clipped = np.minimum(mu_set, w * weight)
                # 3) aggregation = max: รวมหลายกฎเข้าด้วยกันโดยเอาค่าสูงสุดในแต่ละจุด
                agg[out_name] = np.maximum(agg[out_name], clipped)
        # 4) defuzz ทุกเอาต์พุตด้วย centroid
        return {name: defuzz_centroid(var.universe, agg[name]) for name, var in self.outputs.items()}


# -------- โดเมนของตัวแปร (แกน x) --------
u  = np.linspace(0, 100, 1001)   # ใช้กับตัวแปรที่อยู่ใน 0..100 (ละเอียด 1001 จุด)
uR = np.linspace(50, 100, 1001)  # Resolution scale จำกัด 50..100

def mk_perf_sets():
    """ นิยาม MF ของอินพุต CPU/GPU/RAM: low/mid/high """
    return {
        "low":  FuzzySet("low",  lambda x: trapmf(x, (0, 0, 25, 45))),   # ต่ำชัด 0..25, ค่อยๆ เลิกต่ำ 25..45
        "mid":  FuzzySet("mid",  lambda x: trimf(x, (35, 55, 75))),      # กลางพอดีที่ 55
        "high": FuzzySet("high", lambda x: trapmf(x, (65, 80, 100, 100)))# เริ่มสูง ~65, สูงชัด 80..100
    }

# -------- อินพุต 3 ตัว --------
inputs = {
    "cpu": FuzzyVar("cpu", u, mk_perf_sets()),
    "gpu": FuzzyVar("gpu", u, mk_perf_sets()),
    "ram": FuzzyVar("ram", u, mk_perf_sets()),
}

# -------- เอาต์พุต: MF ของ Quality / Resolution / Texture --------
quality_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (0, 0, 25, 40))),
    "med":   FuzzySet("med",   lambda x: trimf(x, (35, 55, 70))),
    "high":  FuzzySet("high",  lambda x: trimf(x, (65, 80, 90))),
    "ultra": FuzzySet("ultra", lambda x: trapmf(x, (85, 92, 100, 100))), # ultra เฉพาะกรณีแรงมาก
}
res_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (50, 50, 60, 75))),   # scale ไม่ต่ำกว่า 50%
    "med":   FuzzySet("med",   lambda x: trimf(x, (70, 80, 90))),
    "high":  FuzzySet("high",  lambda x: trapmf(x, (88, 94, 100, 100))),
}
tex_sets = {
    "low":   FuzzySet("low",   lambda x: trapmf(x, (0, 0, 30, 50))),     # RAM ต่ำ = เท็กซ์เจอร์ต่ำ
    "med":   FuzzySet("med",   lambda x: trimf(x, (45, 60, 75))),
    "high":  FuzzySet("high",  lambda x: trapmf(x, (70, 85, 100, 100))),
}

# -------- ประกาศตัวแปรเอาต์พุต --------
outputs = {
    "quality": FuzzyVar("quality", u,  quality_sets),
    "res":     FuzzyVar("res",     uR, res_sets),
    "tex":     FuzzyVar("tex",     u,  tex_sets),
}

def MU(v, s, x):
    """ ช็อตคัต: μ(ตัวแปร v เป็นเซต s) ที่ค่าอินพุต x[v] """
    return inputs[v].mu(s, x[v])

# -------- กฎแบบภาษาคน (ใช้ใส่รายงาน) --------
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

# -------- กฎใช้งานจริง (AND=min, OR=max, implication=min, aggregation=max) --------
RULES = [
    # R1: ทั้งสามสูง -> คุณภาพรวม/สเกล/เท็กซ์เจอร์ สูงสุด
    Rule(lambda x: min(MU('cpu','high',x), MU('gpu','high',x), MU('ram','high',x)),
         [('quality','ultra',1.0), ('res','high',1.0), ('tex','high',1.0)]),

    # R2: GPU สูง, CPU/RAM กลาง -> คุณภาพ/สเกล สูง, เท็กซ์เจอร์ กลาง
    Rule(lambda x: min(MU('gpu','high',x), MU('cpu','mid',x), MU('ram','mid',x)),
         [('quality','high',1.0), ('res','high',1.0), ('tex','med',1.0)]),

    # R3: ทั้งหมดกลาง -> เอาต์พุตกลาง
    Rule(lambda x: min(MU('cpu','mid',x), MU('gpu','mid',x), MU('ram','mid',x)),
         [('quality','med',1.0), ('res','med',1.0), ('tex','med',1.0)]),

    # R4: GPU ต่ำ หรือ CPU ต่ำ -> กดคุณภาพรวม/สเกล ลง
    Rule(lambda x: max(MU('gpu','low',x), MU('cpu','low',x)),
         [('quality','low',1.0), ('res','low',1.0)]),

    # R5: RAM ต่ำ -> เท็กซ์เจอร์ต่ำ (คอขวดหน่วยความจำ)
    Rule(lambda x: MU('ram','low',x),
         [('tex','low',1.0)]),

    # R6: GPU สูง + RAM กลาง -> ดันคุณภาพ/สเกลขึ้น แต่ใส่ weight 0.9 ให้ไม่ชนกับ R1 แรงเกิน
    Rule(lambda x: min(MU('gpu','high',x), MU('ram','mid',x)),
         [('quality','high',0.9), ('res','high',0.9), ('tex','med',1.0)]),

    # R7: CPU กลาง + GPU สูง + RAM สูง -> เอาต์พุตสูง
    Rule(lambda x: min(MU('cpu','mid',x), MU('gpu','high',x), MU('ram','high',x)),
         [('quality','high',1.0), ('res','high',1.0), ('tex','high',1.0)]),

    # R8: CPU สูง + GPU กลาง + RAM สูง -> Quality สูง (0.8) เพื่อละมุน, Res กลาง, Tex สูง
    Rule(lambda x: min(MU('cpu','high',x), MU('gpu','mid',x), MU('ram','high',x)),
         [('quality','high',0.8), ('res','med',1.0), ('tex','high',1.0)]),

    # R9: GPU กลาง + RAM ต่ำ -> เท็กซ์เจอร์ต่ำ; สเกลมีแนวโน้มต่ำ (0.8)
    Rule(lambda x: min(MU('gpu','mid',x), MU('ram','low',x)),
         [('tex','low',1.0), ('res','low',0.8)]),

    # R10: CPU ต่ำ + GPU สูง -> คุณภาพ/สเกล ระดับกลาง (คอขวด CPU)
    Rule(lambda x: min(MU('cpu','low',x), MU('gpu','high',x)),
         [('quality','med',1.0), ('res','med',1.0)]),
]

# ประกอบระบบ
system = MamdaniSystem(inputs, outputs, RULES)

# -------- Numeric -> Label mapping (ไว้โชว์/รายงาน/ UI) --------
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

# -------- Hysteresis: กันฉลาก "กระพริบ" เมื่อค่าใกล้เส้นแบ่ง --------
class HysteresisMapper:
    """
    levels: รายการฉลาก (เรียงลำดับจากต่ำไปสูง)
    rise: เกณฑ์ "ขยับขึ้นระดับ"
    fall: เกณฑ์ "ขยับลงระดับ"
    """
    def __init__(self, levels, rise, fall, initial=None):
        self.levels, self.rise, self.fall = levels, rise, fall
        self.idx = 0 if initial is None else levels.index(initial)
    def update(self, x: float) -> str:
        # ขึ้นระดับเมื่อถึงเกณฑ์ rise
        while self.idx < len(self.rise) and x >= self.rise[self.idx]:
            self.idx += 1
        # ลงระดับเมื่อค่าต่ำกว่าเกณฑ์ fall
        while self.idx > 0 and x < self.fall[self.idx - 1]:
            self.idx -= 1
        return self.levels[self.idx]

quality_mapper = HysteresisMapper(["Low","Medium","High","Ultra"], [40.0, 70.0, 90.0], [35.0, 65.0, 85.0])
res_mapper     = HysteresisMapper(["Low","Medium","High"],          [75.0, 90.0],       [70.0, 88.0])
tex_mapper     = HysteresisMapper(["Low","Medium","High"],          [50.0, 75.0],       [45.0, 70.0])

# -------- APIs หลักสำหรับเรียกใช้งาน --------
def infer(cpu: float, gpu: float, ram: float) -> Dict[str, float]:
    """ คืนค่าเอาต์พุตแบบตัวเลข (crisp) ทั้ง 3 ตัว จากอินพุต cpu/gpu/ram """
    return system.infer({"cpu": cpu, "gpu": gpu, "ram": ram})

def infer_with_labels(cpu: float, gpu: float, ram: float, use_hysteresis: bool = False) -> Dict[str, object]:
    """ คืนทั้งค่าเลข + ฉลาก (เลือกใช้ hysteresis ได้) — เหมาะกับ UI/รายงาน """
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
        "quality_idx": round(out["quality"], 2), "quality_lbl": qlbl,
        "res_scale_%": round(out["res"], 1),     "res_label": rlbl,
        "texture_idx": round(out["tex"], 1),     "texture_lbl": tlbl,
    }

def get_rules_text() -> str:
    """ รวมกฎแบบภาษาคนไว้ใช้แสดงในรายงาน/คอนโซล """
    return "\n".join(RULES_HUMAN)

# -------- Utilities วาดกราฟ (สำหรับรายงาน) --------
def plot_memberships(var, title, filename=None):
    """ วาดกราฟ MF ของตัวแปรหนึ่งตัว (อินพุตหรือเอาต์พุต) """
    plt.figure()
    for name, fset in var.sets.items():
        plt.plot(var.universe, fset.mf(var.universe), label=name)
    plt.title(title)
    plt.xlabel(var.name)
    plt.ylabel("μ")
    plt.legend(loc="best")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=160)
        plt.close()
    else:
        plt.show()

def demo_defuzz(cpu=70, gpu=80, ram=65, out_name="quality", filename=None):
    """ เดโม: โชว์ขั้น clip/aggregate และ centroid ของเอาต์พุตชื่อ out_name """
    x = {"cpu": cpu, "gpu": gpu, "ram": ram}
    # รวมผลหลายกฎ (เฉพาะเอาต์พุตที่สนใจ) ลงในกราฟ agg
    agg = np.zeros_like(outputs[out_name].universe, dtype=float)
    for rule in RULES:
        w = float(rule.antecedent(x))
        if w <= 0:
            continue
        for o_name, set_name, weight in rule.consequents:
            if o_name != out_name:
                continue
            var = outputs[o_name]
            mu = var.sets[set_name].mf(var.universe)
            clipped = np.minimum(mu, w * weight)   # implication=min
            agg = np.maximum(agg, clipped)         # aggregation=max

    xs = outputs[out_name].universe
    c = defuzz_centroid(xs, agg)                  # centroid = คำตอบแบบตัวเลข

    # วาดทั้งเซตพื้นฐาน + กราฟ aggregated + เส้น centroid
    plt.figure()
    for name, fset in outputs[out_name].sets.items():
        plt.plot(xs, fset.mf(xs), label=name)
    plt.plot(xs, agg, label="aggregated")
    plt.axvline(c, label=f"centroid={c:.2f}")
    plt.title(f"Defuzzification demo ({out_name}) @ CPU={cpu}, GPU={gpu}, RAM={ram}")
    plt.xlabel(out_name)
    plt.ylabel("μ")
    plt.legend(loc="best")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=160)
        plt.close()
    else:
        plt.show()
    return c

def plot_heatmap(x_name, y_name, out_name, fixed, filename=None):
    """ Heatmap เอาต์พุต out_name บนระนาบอินพุต 2 ตัว (อีกตัวตรึงตาม fixed) """
    nx = 80; ny = 80  # ปรับความละเอียดได้
    xs = np.linspace(0, 100, nx)
    ys = np.linspace(0, 100, ny)
    Z = np.zeros((ny, nx), dtype=float)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            sample = {**fixed, x_name: float(xv), y_name: float(yv)}
            val = infer(sample["cpu"], sample["gpu"], sample["ram"])[out_name]
            Z[j, i] = val
    plt.figure()
    plt.imshow(Z, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], aspect="auto")
    plt.colorbar(label=f"{out_name} (crisp)")
    plt.xlabel(x_name.upper())
    plt.ylabel(y_name.upper())
    fixed_txt = ", ".join(f"{k.upper()}={v}" for k,v in fixed.items())
    plt.title(f"{out_name} heatmap ({x_name.upper()}×{y_name.upper()} | {fixed_txt})")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=160)
        plt.close()
    else:
        plt.show()

def sensitivity_curve(vary, fixed, out_names=("quality","res","tex"), filename=None):
    """ ไล่ค่าตัวแปรหนึ่งตัว (vary) เพื่อดูความไวของเอาต์พุตที่เลือก """
    xs = np.linspace(0,100,101)
    plt.figure()
    for out_name in out_names:
        ys = []
        for xval in xs:
            sample = {**fixed, vary: float(xval)}
            out = infer(sample["cpu"], sample["gpu"], sample["ram"])
            ys.append(out[out_name])
        plt.plot(xs, ys, label=out_name)
    plt.title(f"Sensitivity: vary {vary.upper()} | fixed " + ", ".join(f"{k.upper()}={v}" for k,v in fixed.items()))
    plt.xlabel(vary.upper())
    plt.ylabel("crisp output")
    plt.legend(loc="best")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=160)
        plt.close()
    else:
        plt.show()

def random_stats(n=300, csv_path="fuzzy_sim_results.csv", fig_hist=None, fig_bar=None):
    """ สุ่มอินพุต n เคส -> เก็บผล/ฉลากเป็น CSV + วาดฮิสโตแกรม/แท่งสัดส่วนฉลาก """
    rng = np.random.default_rng(42)  # seed เดิมเพื่อทำซ้ำ
    rows = []
    lblQ = {"Low":0,"Medium":0,"High":0,"Ultra":0}
    lblR = {"Low":0,"Medium":0,"High":0}
    lblT = {"Low":0,"Medium":0,"High":0}
    Q, R, T = [], [], []
    for _ in range(n):
        cpu = float(rng.uniform(5,95))
        gpu = float(rng.uniform(5,95))
        ram = float(rng.uniform(5,95))
        out = infer_with_labels(cpu, gpu, ram, use_hysteresis=False)
        rows.append(out)
        Q.append(out["quality_idx"]); lblQ[out["quality_lbl"]] += 1
        R.append(out["res_scale_%"]); lblR[out["res_label"]] += 1
        T.append(out["texture_idx"]); lblT[out["texture_lbl"]] += 1

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Histogram of Quality
    plt.figure()
    plt.hist(Q, bins=15)
    plt.title("Histogram of Quality Index")
    plt.xlabel("Quality (0–100)")
    plt.ylabel("count")
    plt.tight_layout()
    if fig_hist:
        plt.savefig(fig_hist, dpi=160)
        plt.close()
    else:
        plt.show()

    # Bar: Quality label distribution
    plt.figure()
    xs = list(lblQ.keys()); ys = [lblQ[k] for k in xs]
    plt.bar(xs, ys)
    plt.title("Quality label distribution")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.tight_layout()
    if fig_bar:
        plt.savefig(fig_bar, dpi=160)
        plt.close()
    else:
        plt.show()

    return df

# -------- Main: ตัวอย่างรัน + สร้างรูปสำหรับรายงาน --------
if __name__ == "__main__":
    outdir = os.getcwd()

    # ตัวอย่างอนุมาน 1 ชุด + แสดงกฎภาษาคน
    print(infer_with_labels(90, 40, 95))
    print("--- Rules ---")
    print(get_rules_text())

    # 1) Memberships 
    plot_memberships(inputs["cpu"], "CPU score memberships", os.path.join(outdir, "fig_membership_cpu.png"))
    plot_memberships(inputs["gpu"], "GPU score memberships", os.path.join(outdir, "fig_membership_gpu.png"))
    plot_memberships(inputs["ram"], "RAM score memberships", os.path.join(outdir, "fig_membership_ram.png"))
    plot_memberships(outputs["quality"], "Output: Graphics Quality memberships", os.path.join(outdir, "fig_membership_quality.png"))
    plot_memberships(outputs["res"], "Output: Resolution Scale memberships", os.path.join(outdir, "fig_membership_res.png"))
    plot_memberships(outputs["tex"], "Output: Texture Quality memberships", os.path.join(outdir, "fig_membership_tex.png"))

    # 2) Defuzz demo 
    demo_defuzz(70, 80, 65, "quality", os.path.join(outdir, "fig_defuzz_quality_demo.png"))

    # 3) Heatmaps 
    plot_heatmap("cpu","gpu","quality", fixed={"cpu":0,"gpu":0,"ram":70}, filename=os.path.join(outdir, "fig_heatmap_quality_cpu_gpu_ram70.png"))
    plot_heatmap("gpu","ram","res", fixed={"cpu":70,"gpu":0,"ram":0}, filename=os.path.join(outdir, "fig_heatmap_res_gpu_ram_cpu70.png"))
    plot_heatmap("ram","gpu","tex", fixed={"cpu":70,"gpu":0,"ram":0}, filename=os.path.join(outdir, "fig_heatmap_tex_ram_gpu_cpu70.png"))

    # 4) Sensitivity 
    sensitivity_curve("gpu", {"cpu":70,"gpu":0,"ram":70},
                      filename=os.path.join(outdir, "fig_sensitivity_vary_gpu_fix_cpu70_ram70.png"))

    # 5) Random stats + CSV 
    random_stats(n=300,
                 csv_path=os.path.join(outdir, "fuzzy_sim_results.csv"),
                 fig_hist=os.path.join(outdir, "fig_hist_quality.png"),
                 fig_bar=os.path.join(outdir, "fig_bar_quality_labels.png"))
