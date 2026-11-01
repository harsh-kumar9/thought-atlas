"""Build a hashed review companion without raw traces, activations, or model weights."""
from pathlib import Path
import hashlib,json,zipfile
P=Path(__file__).resolve().parents[1]
files=set()
for directory in ['supplement','scripts','revision/scripts','revision/tests','revision/protocols']:
 files.update(p for p in (P/directory).rglob('*') if p.is_file() and p.suffix in {'.py','.sbatch','.json','.csv','.txt'})
for directory in ['revision/results','data/figure_inputs']:
 files.update(p for p in (P/directory).rglob('*') if p.is_file() and p.suffix in {'.csv','.json'})
files.update((P/'revision/sources').rglob('*.py'))
files.update((P/'revision/sources/vectors').glob('*.npz'))
files.update([P/'revision/STATUS.json',P/'revision/REVIEW_RESPONSE.json',P/'figure_manifest.json',P/'model_names.json',P/'figure_style.json'])
files=sorted(f for f in files if '__pycache__' not in str(f))
manifest={
 'scope':'Review evidence companion, not an anonymized submission or a self-contained GPU rerun environment.',
 'included':'Aggregate analysis outputs, exact prompts, frozen confirmation protocol, code, inspected historical source snapshots, and small steering/readout parameter arrays.',
 'excluded':'Raw full trace releases, activation-fitting arrays, pretrained model weights, credentials, environments, and unfinished GPU outputs.',
 'reproduction':'Figure and analysis scripts retain repository-relative dependencies. Source input hashes identify omitted large artifacts on the recorded research environment. Frozen protocol files state which results remain pending.',
 'files':[{'path':str(f.relative_to(P)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]}
out=P/'thought-atlas-evidence.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,f.relative_to(P))
 z.writestr('EVIDENCE_MANIFEST.json',json.dumps(manifest,indent=2))
(P/'evidence_manifest.json').write_text(json.dumps(dict(archive=out.name,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),**manifest),indent=2)+'\n')
print(f'{len(files)} files; {out.stat().st_size} bytes')
