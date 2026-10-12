"""Aggregate completed evaluations without merging unlike scientific protocols."""
from pathlib import Path
import argparse, csv, json, statistics

ap=argparse.ArgumentParser()
ap.add_argument('--runs',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
rows=[]
scientific=['dataset','backbone','method','loss','alpha','rho','gamma','gamma_semantics',
    'K','lr','grid','posterior_scale','base_scale','gradient_norm_matching','optimizer',
    'protocol','num_examples','severity','batch_size','steps','corruptions','checkpoint_sha256','cfg_overrides','env']
metrics=['accuracy_percent','mmc','ece','nll']
groups={}
for p in sorted(args.runs.glob('*/result.json')):
    r=json.loads(p.read_text())
    if r.get('status')!='complete':continue
    key=json.dumps({k:r.get(k) for k in scientific},sort_keys=True)
    groups.setdefault(key,[]).append(r)
    rows.append({k:r.get(k) for k in ['run_id','seed',*scientific,*metrics]})

def write(name,records):
    if not records:return
    with (args.output/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

write('seed_results.csv',rows)
aggregates=[]
for key,runs in groups.items():
    seeds=[r['seed'] for r in runs]
    if len(set(seeds))!=len(seeds):raise SystemExit(f'Duplicate seed in a scientific group: {seeds}')
    row=dict(json.loads(key),n_seeds=len(seeds),seeds=','.join(map(str,sorted(seeds))),
             sd_definition='sample SD, ddof=1; accuracy SD is in percentage points')
    for metric in metrics:
        values=[r[metric] for r in runs]
        row[metric+'_mean']=statistics.mean(values)
        row[metric+'_sd']=statistics.stdev(values) if len(values)>1 else ''
    aggregates.append(row)
write('mean_sd_summary.csv',aggregates)
print(f'Saved {len(rows)} complete runs in {len(aggregates)} separate scientific groups')
