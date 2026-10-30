import os, sys
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
os.chdir(ROOT)
import glob, json, random, sys
sys.path[:0] = [str(ROOT/'ALFWorld'), str(ROOT/'experiments/alfworld_baselines')]
from types import SimpleNamespace
from methods import judge
cfg = SimpleNamespace(llm_model='qwen3-32b', llm_base_url='http://localhost:8002/v1')
fs = sorted(glob.glob(str(ROOT)+'/experiments/alfworld_ace_sd/runs/20261029_iter/beta_1/*/results/*.json'))
random.Random(0).shuffle(fs)
tp = fp = tn = fn = 0; miss = 0
for f in fs[:150]:
    x = json.load(open(f))
    v = judge(cfg, x['goal'], x['trajectory_text']); miss += bool(v['label_missing'])
    y, j = x['success'], v['judged_success']
    tp += y and j; fn += y and not j; fp += (not y) and j; tn += (not y) and not j
print(f'n=150 acc={(tp+tn)/150:.3f} TP={tp} FN={fn} FP={fp} TN={tn} precision={tp/max(tp+fp,1):.3f} recall={tp/max(tp+fn,1):.3f} fail-recall={tn/max(tn+fp,1):.3f} label_missing={miss}')
