"""Run frozen TENT/MIRA code, retaining domain results and label-free batch diagnostics."""
from pathlib import Path
import argparse, csv, fcntl, hashlib, json, os, random, subprocess, sys, time, types

REPO=Path(os.environ.get('MIRA_REPO_ROOT',str(Path(__file__).resolve().parents[1])))
HERE=Path(os.environ.get('MIRA_RESULTS_ROOT',str(REPO/'runs'))).resolve()
ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--max-batches',type=int,default=0)
args=ap.parse_args(); raw=os.path.expandvars(Path(args.config).read_text()); assert '${' not in raw,'Set MIRA_DATA_ROOT and MIRA_CHECKPOINT_ROOT before running'; spec=json.loads(raw)
source_manifest=json.loads((REPO/'reference_manifest.json').read_text())
for item in source_manifest['frozen_files']+source_manifest.get('api_and_entrypoint_files',[])+source_manifest.get('config_files',[]):
    path=REPO/item['path']
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],'Recorded source/config changed: '+item['path']
checkpoint_digest=hashlib.sha256(Path(spec['checkpoint']).read_bytes()).hexdigest()
assert checkpoint_digest==spec['checkpoint_sha256'],'Source checkpoint SHA256 mismatch'
run=HERE/spec['run_id']; run.mkdir(parents=True,exist_ok=True)
scientific=['cfg_overrides','checkpoint_sha256','env','severity','batch_size','steps','stream_seed','perturbation_seed','dataset','backbone','seed','method','loss','alpha','rho','gamma','gamma_semantics','K','lr','grid','posterior_scale','base_scale','gradient_norm_matching','gradient_norm_target','corruptions','data_dir','checkpoint']
if (run/'result.json').exists():
    previous=json.loads((run/'result.json').read_text())
    mismatch=[k for k in scientific if previous.get(k)!=spec.get(k)]
    for k,default in [('optimizer','SGD'),('num_examples',10000)]:
        if previous.get(k,default)!=spec.get(k,default): mismatch.append(k)
    assert not mismatch,'Completed result differs from requested config: '+','.join(mismatch)
    print('Already complete with matching scientific config:',run,flush=True); sys.exit(0)
if (run/'config.json').exists():
    previous=json.loads((run/'config.json').read_text())
    mismatch=[k for k in scientific if previous.get(k)!=spec.get(k)]
    for k,default in [('optimizer','SGD'),('num_examples',10000),('gradient_norm_target',1.0)]:
        if previous.get(k,default)!=spec.get(k,default): mismatch.append(k)
    assert not mismatch,'Partial run differs from requested scientific config: '+','.join(mismatch)
if Path(__file__).resolve().parent!=run:
    source=Path(__file__).read_bytes(); frozen=run/('worker_'+hashlib.sha256(source).hexdigest()[:16]+'.py')
    if not frozen.exists(): frozen.write_bytes(source)
    env=os.environ.copy(); env['MIRA_RESULTS_ROOT']=str(HERE); env['MIRA_REPO_ROOT']=str(REPO)
    os.execve(sys.executable,[sys.executable,str(frozen),*sys.argv[1:]],env)
visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
# GPU selection belongs to the caller, not to scientific configuration.
gpu_lock=(HERE/('.gpu-'+visible.replace('/','_')+'.lock')).open('w'); fcntl.flock(gpu_lock,fcntl.LOCK_EX)
run_start=time.monotonic()
for key in list(os.environ):
    if key.startswith(('BETTA_','TTA_GRAD_')): os.environ.pop(key)
os.environ.update(spec['env'])
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(HERE/'cache/xdg'))
snapshot=REPO/'reference_runtime'; sys.path.insert(0,str(REPO)); sys.path.insert(0,str(snapshot)); os.chdir(snapshot)
# Import exactly the requested historical methods; avoid importing unrelated TTA plug-ins.
pkg=types.ModuleType('methods'); pkg.__path__=[str(snapshot/'methods')]; sys.modules['methods']=pkg
import numpy as np
import torch
from conf import cfg as original_cfg
from models.model import get_model
import models.model as model_module
from mira.checkpoints import install_local_torchvision_loader
install_local_torchvision_loader(model_module, spec['checkpoint'])
from datasets.data_loading import get_test_loader
from methods.tent import Tent
from methods.tent2 import Tent2
from methods.source import Source
from methods.tent_laplace_betta import Tent_laplace_betta
from methods.betta_core import find_last_linear

assert torch.cuda.is_available(), 'CUDA unavailable; CPU results must not be silently substituted.'
torch.set_num_threads(4); torch.hub.set_dir(str(HERE/'cache/hub'))
seed=spec['seed']; stream_seed=spec.get('stream_seed',seed); perturbation_seed=spec.get('perturbation_seed',seed)
random.seed(stream_seed); np.random.seed(stream_seed); torch.manual_seed(seed); torch.cuda.manual_seed(perturbation_seed)
assert spec.get('batch_size',64) in {64,128} and spec.get('steps',1)==1 and spec.get('severity',5)==5,'Reference runner requires explicit batch64/128, step1 and severity5'
torch.backends.cudnn.benchmark=True; torch.backends.cudnn.deterministic=False
cfg=original_cfg.clone(); cfg.defrost()
cfg.MODEL.ARCH=spec['backbone']; cfg.MODEL.ADAPTATION=spec['method']; cfg.MODEL.USE_CLIP=False
cfg.MODEL.EPISODIC=False; cfg.MODEL.RESET_AFTER_NUM_UPDATES=0; cfg.MODEL.CKPT_PATH=''
cfg.MODEL.WEIGHTS='IMAGENET1K_V1'; cfg.CKPT_DIR=spec['checkpoint_dir']; cfg.DATA_DIR=spec['data_dir']
cfg.CORRUPTION.DATASET=spec['dataset']; cfg.CORRUPTION.TYPE=spec['corruptions']; cfg.CORRUPTION.SEVERITY=[5]; cfg.CORRUPTION.NUM_EX=spec.get('num_examples',10000)
cfg.RNG_SEED=seed; cfg.SETTING='reset_each_shift'; cfg.MIXED_PRECISION=False; cfg.DETERMINISM=False
cfg.NUM_MC=spec['K']
cfg.OPTIM.METHOD=spec.get('optimizer','SGD'); cfg.OPTIM.LR=spec['lr']; cfg.OPTIM.MOMENTUM=0.9; cfg.OPTIM.DAMPENING=0.0
cfg.OPTIM.NESTEROV=True; cfg.OPTIM.WD=0.0; cfg.OPTIM.STEPS=1; cfg.OPTIM.BETA=0.9
cfg.TEST.BATCH_SIZE=spec.get('batch_size',64); cfg.TEST.NUM_WORKERS=4; cfg.TEST.WINDOW_LENGTH=1; cfg.TEST.LANDSCAPE=False
cfg.TEST.CHECK_UPDATE=False; cfg.TEST.ENTROPY_TEMPERATURE=1.0
if spec.get('cfg_overrides'): cfg.merge_from_other_cfg(type(cfg)(spec['cfg_overrides']))
cfg.freeze()
(run/'resolved.yaml').write_text(cfg.dump()); (run/'config.json').write_text(json.dumps(spec,indent=2)+'\n')
print('START',spec['run_id'],'GPU',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True)
base,preprocess=get_model(cfg,100 if spec['dataset']=='cifar100_c' else 10 if spec['dataset']=='cifar10_c' else 1000,'cuda')
base.model_preprocess=preprocess
import inspect
metadata=dict(backbone=spec['backbone'],checkpoint_sha256=checkpoint_digest,
    attention_fused=[bool(m.fused_attn) for m in base.modules() if hasattr(m,'fused_attn')],
    layernorm_eps=sorted({float(m.eps) for m in base.modules() if isinstance(m,torch.nn.LayerNorm)}),
    normalization=[dict(name=n,mean=m.mean.detach().cpu().flatten().tolist(),std=m.std.detach().cpu().flatten().tolist())
        for n,m in base.named_modules() if isinstance(getattr(m,'mean',None),torch.Tensor) and isinstance(getattr(m,'std',None),torch.Tensor)],
    prediction_dtype=str(next(base.parameters()).dtype),
    imagenet_c_preprocessing='already224x224; PIL RGB and ToTensor; normalization in model')
if spec['backbone'].startswith('vit_base_'):
    import timm
    model_module_source=Path(inspect.getfile(timm.models.vision_transformer))
    metadata.update(timm_version=timm.__version__,timm_vit_source_sha256=hashlib.sha256(model_module_source.read_bytes()).hexdigest())
(run/'model_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
classes=100 if spec['dataset']=='cifar100_c' else 10 if spec['dataset']=='cifar10_c' else 1000
method_types={'tent':Tent,'tent2':Tent2,'source':Source,'tent_laplace_betta':Tent_laplace_betta}
method=method_types[spec['method']](cfg,base,classes)
if spec['method'].endswith('laplace_betta'):
    effective=dict(K=method.K,alpha=method.mi_elbow_alpha,rho=method.agreement_threshold,
        tau=method.loss_tau,gamma=method.loss_gamma,loss_type=method.loss_type,
        selector=method.selection,same_noise=method.same_noise,interpolation=method.sigma_interpolate,
        posterior_scale=method.model.posterior_scale_mode,base_scale=method.sigma_scale_mode,
        bias_scale=method.bias_scale_mode,grad_norm_match=method.grad_norm_match,
        grad_norm_target=method.grad_norm_target,controlled_objective=spec['gamma_semantics'])
    assert effective['K']==spec['K'] and effective['alpha']==spec['alpha'] and effective['rho']==spec['rho']
    assert effective['same_noise'] and effective['interpolation']
    assert effective['posterior_scale']==spec['posterior_scale'] and effective['base_scale']==spec['base_scale']
    (run/'mira_effective_settings.json').write_text(json.dumps(effective,indent=2)+'\n')
classifier=find_last_linear(method.model); fixed_w=classifier.weight.detach().clone(); fixed_b=None if classifier.bias is None else classifier.bias.detach().clone()
checks={'classifier_unchanged':True,'sample_weights_detached':True,'finite_gradients':True,'source_reset_verified':True}
parameter_names=[n for n,p in method.model.named_parameters() if p.requires_grad]
(run/'adaptation_parameters.json').write_text(json.dumps(parameter_names,indent=2))
manifest=json.loads((REPO/'reference_manifest.json').read_text())
public_commit=None; public_dirty=None
try:
    top=subprocess.check_output(['git','-C',str(REPO),'rev-parse','--show-toplevel'],stderr=subprocess.DEVNULL,text=True).strip()
    if REPO.resolve() in {Path(top).resolve(),(Path(top)/'code').resolve()}:
        public_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
        public_dirty=bool(subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip())
except (subprocess.CalledProcessError,FileNotFoundError):
    pass
(run/'provenance.json').write_text(json.dumps(dict(code_commit=manifest['code_commit'],manifest_sha256=hashlib.sha256((REPO/'reference_manifest.json').read_bytes()).hexdigest(),
 public_repository_commit=public_commit,public_working_tree_modified=public_dirty,
 python=sys.executable,worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu_name=torch.cuda.get_device_name(0),
 gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),stream_rng={'python':stream_seed,'numpy':stream_seed,'selection':'historical random.shuffle'+(' then random.sample' if spec.get('num_examples',10000)>0 else ' only')},
 perturbation_rng={'torch_cuda':perturbation_seed,'shared_across_candidate_sigmas':True,'across_runs':'same CUDA seed and draw shape for compared MIRA methods'},
 cpu_torch_rng=seed,evaluation=f"pre-update prediction; macro-average over {len(spec['corruptions'])} corruption metrics; severity5; ECE10 equal-width bins",
 checkpoint=spec['checkpoint'],parameter_count=sum(p.numel() for p in method.params),checkpoints=manifest['checkpoints']),indent=2))
if spec.get('gamma_semantics')=='EH_coefficient_only' or spec.get('loss')=='confidence_only':
    def controlled_objective(self,HE,EH=None):
        betta=self._last_betta_out; H=betta['HE']; E=betta['EH']; confidence=betta['p_mean'].max(-1).values
        wc=self._normalize_loss_weights(confidence.clamp_min(1e-12))
        w=wc if spec.get('loss')=='confidence_only' else self._normalize_loss_weights(self._low_score_weight(H-E)*wc)
        return (w*(H+spec['gamma']*E)).mean()
    method._betta_objective=types.MethodType(controlled_objective,method)
if hasattr(method,'_normalize_loss_weights'):
    normalizer=method._normalize_loss_weights
    def checked_normalizer(w,*a,**kw):
        result=normalizer(w,*a,**kw); checks['sample_weights_detached'] &= not result.requires_grad
        assert not result.requires_grad, 'Unexpected gradient through sample weights'
        return result
    method._normalize_loss_weights=checked_normalizer
last_gradient={}; monitored_optimizer=getattr(method.optimizer,'base_optimizer',method.optimizer); optimizer_step=monitored_optimizer.step if monitored_optimizer is not None else None
def checked_step(*a,**kw):
    norm=torch.zeros((),device='cuda')
    for p in method.params:
        if p.grad is not None: norm+=p.grad.detach().float().square().sum()
    val=float(norm.sqrt()); checks['finite_gradients'] &= bool(np.isfinite(val))
    assert np.isfinite(val),'Nonfinite adaptation gradient'
    raw=float(getattr(method,'last_raw_grad_norm',val)) if getattr(method,'grad_norm_match',False) else val
    last_gradient.update(actual_grad_norm=val,raw_grad_norm=raw,grad_scale=float(getattr(method,'last_grad_norm_scale',1.0)))
    return optimizer_step(*a,**kw)
if monitored_optimizer is not None: monitored_optimizer.step=checked_step

def metrics(prob,target):
    confidence,pred=prob.max(-1); correct=(pred==target).float()
    ece=0.0
    for i in range(10):
        mask=(confidence>=i/10)&((confidence< (i+1)/10) if i<9 else (confidence<=1))
        if mask.any(): ece+=float(mask.float().mean()*abs(confidence[mask].mean()-correct[mask].mean()))
    return dict(accuracy_percent=100*float(correct.mean()),mmc=float(confidence.mean()),ece=ece,nll=float(-prob.clamp_min(1e-12).log()[torch.arange(len(target)),target.long()].mean()))
def convert(x):
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    if isinstance(x,(np.floating,np.integer)): return x.item()
    return x

start=time.monotonic(); domain_rows=[]
for domain in spec['corruptions']:
    done=run/(domain+'.json'); rng_path=run/(domain+'.rng.pt')
    if done.exists() and rng_path.exists() and not args.max_batches:
        domain_rows.append(json.loads(done.read_text())); rng=torch.load(rng_path,weights_only=False)
        torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda']); continue
    method.reset()
    for m,state in zip(method.models,method.model_states):
        checks['source_reset_verified'] &= all(torch.equal(v,state[k]) for k,v in m.state_dict().items())
    assert checks['source_reset_verified'],'Run did not reset to source'
    loader=get_test_loader(setting=cfg.SETTING,adaptation=cfg.MODEL.ADAPTATION,dataset_name=cfg.CORRUPTION.DATASET,
      preprocess=preprocess,data_root_dir=cfg.DATA_DIR,domain_name=domain,domain_names_all=cfg.CORRUPTION.TYPE,severity=5,
      num_examples=cfg.CORRUPTION.NUM_EX,rng_seed=stream_seed,use_clip=False,batch_size=cfg.TEST.BATCH_SIZE,shuffle=False,workers=4)
    # Digest ordered target inputs/identifiers without using labels for selection.
    sh=hashlib.sha256(); portable_sh=hashlib.sha256()
    for sample in loader.dataset.samples:
        value=sample[0]; sh.update(value.encode() if isinstance(value,str) else np.asarray(value).tobytes())
        if isinstance(value,str):
            identifier=Path(value).absolute().relative_to(Path(cfg.DATA_DIR).absolute()).as_posix().encode()
        else:
            identifier=np.asarray(value).tobytes()
        portable_sh.update(len(identifier).to_bytes(8,'big')); portable_sh.update(identifier)
    target_stream_hash=sh.hexdigest(); portable_stream_hash=portable_sh.hexdigest()
    all_p=[]; all_y=[]; elapsed_compute=0.; batch_rows=[]
    torch.cuda.reset_peak_memory_stats(); domain_start=time.monotonic()
    with (run/(domain+'.batches.jsonl')).open('w') as trace:
        for b_idx,batch in enumerate(loader):
            imgs=batch[0].cuda(non_blocking=True); y=batch[1].long().cpu()
            rng_before=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest() if b_idx==0 else None
            torch.cuda.synchronize(); tick=time.monotonic()
            last_gradient.clear()
            out=method([imgs]); torch.cuda.synchronize(); elapsed_compute+=time.monotonic()-tick
            prob=out.detach().softmax(-1).cpu(); all_p.append(prob); all_y.append(y)
            if not torch.equal(classifier.weight.detach(),fixed_w) or (fixed_b is not None and not torch.equal(classifier.bias.detach(),fixed_b)):
                checks['classifier_unchanged']=False; raise AssertionError('Classifier was not restored')
            row=dict(domain=domain,batch=b_idx,batch_size=len(y),**last_gradient)
            if rng_before is not None: row['cuda_rng_before_sha256']=rng_before
            if hasattr(method,'scale_trace') and method.scale_trace:
                stats=method.scale_trace[-1]
                row.update({k:convert(v) for k,v in stats.items()})
                row['sigma']=row['selected_sigma']
                row['sigma_fractional_idx']=row['selected_sigma_fractional_idx']
                mi=np.asarray(row.get('grid_MI',[]),dtype=np.float32); ag=np.asarray(row.get('grid_agreement',[]),dtype=np.float32)
                target=float(mi.min()+np.float32(spec['alpha'])*(mi.max()-mi.min()))
                feasible=(mi>=target)&(ag>=np.float32(spec['rho']))
                anchor=int(row['selected_idx_anchor'])
                row.update(mi_threshold=target,mi_min=float(mi.min()),mi_max=float(mi.max()),mi_span=float(mi.max()-mi.min()),
                    selected_anchor_mi=float(mi[anchor]),selected_anchor_agreement=float(ag[anchor]),
                    no_feasible=not bool(feasible.any()),fallback=not bool(feasible.any()))
                frac=float(row.get('sigma_fractional_idx',row.get('sigma_idx',0)))
                row.update(lower_boundary=frac<=1.0+1e-7,upper_boundary=frac>=len(row['grid_sigmas'])-2-1e-7)
                # No history is required by the algorithm; traces are persisted immediately.
                method.scale_trace.clear(); method.sigma_log.clear()
            trace.write(json.dumps(row,allow_nan=True)+'\n'); trace.flush(); batch_rows.append(row)
            if b_idx%50==0: print(domain,b_idx,'/',len(loader),flush=True)
            if args.max_batches and b_idx+1>=args.max_batches: break
    m=metrics(torch.cat(all_p),torch.cat(all_y))
    row=dict(domain=domain,n_samples=sum(len(y) for y in all_y),n_batches=len(batch_rows),target_stream_sha256=target_stream_hash,
       ordered_image_ids_sha256=portable_stream_hash,ordered_image_ids_definition='data-root-relative identifiers with 8-byte length prefixes; input order preserved',
       runtime_compute_seconds=elapsed_compute,runtime_wall_seconds=time.monotonic()-domain_start,
       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,**m)
    if batch_rows and 'sigma' in batch_rows[0]:
        vals=np.asarray([r['sigma'] for r in batch_rows]); row.update(sigma_mean=float(vals.mean()),sigma_median=float(np.median(vals)),sigma_min=float(vals.min()),sigma_max=float(vals.max()),
        no_feasible_ratio=float(np.mean([r['no_feasible'] for r in batch_rows])),fallback_ratio=float(np.mean([r['fallback'] for r in batch_rows])),
        lower_boundary_ratio=float(np.mean([r['lower_boundary'] for r in batch_rows])),upper_boundary_ratio=float(np.mean([r['upper_boundary'] for r in batch_rows])))
    domain_rows.append(row)
    if not args.max_batches:
        torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()),rng_path)
        done.write_text(json.dumps(row,indent=2)+'\n')
    print('DOMAIN_RESULT',json.dumps(row),flush=True)
    if args.max_batches: break

keys=['accuracy_percent','mmc','ece','nll']
result={**spec,'status':'smoke_only' if args.max_batches else 'complete','domains_completed':len(domain_rows),
        **{k:float(np.mean([r[k] for r in domain_rows])) for k in keys},'checks':checks,
        'runtime_compute_seconds':sum(r['runtime_compute_seconds'] for r in domain_rows),
        'runtime_wall_seconds':sum(r['runtime_wall_seconds'] for r in domain_rows),
        'runtime_end_to_end_seconds':time.monotonic()-run_start,
        'num_batches':sum(r['n_batches'] for r in domain_rows),
        'peak_allocated_mib':max(r['peak_allocated_mib'] for r in domain_rows),
        'peak_reserved_mib':max(r['peak_reserved_mib'] for r in domain_rows)}
if len(domain_rows)>1:
    result['peak_allocated_after_first_corruption_mib']=max(r['peak_allocated_mib'] for r in domain_rows[1:])
    result['peak_reserved_after_first_corruption_mib']=max(r['peak_reserved_mib'] for r in domain_rows[1:])
sigmas=[]; diagnostic_rows=[]
for domain in [r['domain'] for r in domain_rows]:
    for line in (run/(domain+'.batches.jsonl')).read_text().splitlines():
        r=json.loads(line)
        if 'sigma' in r: sigmas.append(r['sigma']); diagnostic_rows.append(r)
if sigmas:
    result.update(sigma_mean=float(np.mean(sigmas)),sigma_sd=float(np.std(sigmas,ddof=1)),sigma_min=min(sigmas),sigma_max=max(sigmas),
                  sigma_quantiles=dict(zip(['p05','p25','p50','p75','p95'],np.quantile(sigmas,[.05,.25,.5,.75,.95]).tolist())))
    for key in ['no_feasible','fallback','lower_boundary','upper_boundary']:
        result[key+'_ratio']=float(np.mean([r[key] for r in diagnostic_rows]))
all_batches=[]
for domain in [r['domain'] for r in domain_rows]:
    all_batches.extend(json.loads(line) for line in (run/(domain+'.batches.jsonl')).read_text().splitlines())
grad_values=[r['actual_grad_norm'] for r in all_batches if 'actual_grad_norm' in r]
if grad_values:
    result['gradient_norm_summary']=dict(min=min(grad_values),median=float(np.median(grad_values)),max=max(grad_values),
        matching_target=spec.get('gradient_norm_target'),zero_or_nonfinite_batches=sum(not np.isfinite(v) or v<=1e-12 for v in grad_values))
batch_cols=['domain','batch','batch_size','sigma','selected_idx_anchor','selected_idx_nearest','sigma_fractional_idx','sigma_interpolated','sigma_interp_kind','HE_mean','EH_mean','MI_mean','agreement_mean','mi_threshold','mi_min','mi_max','mi_span','selected_anchor_mi','selected_anchor_agreement','no_feasible','fallback','lower_boundary','upper_boundary','actual_grad_norm','raw_grad_norm','grad_scale','cuda_rng_before_sha256']
with (run/'batch_diagnostics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=batch_cols,extrasaction='ignore'); writer.writeheader(); writer.writerows(all_batches)
with (run/'per_corruption.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(domain_rows[0])); writer.writeheader(); writer.writerows(domain_rows)
(run/('smoke_result.json' if args.max_batches else 'result.json')).write_text(json.dumps(result,indent=2)+'\n')
print('RESULT',json.dumps({k:result[k] for k in keys}),flush=True)
