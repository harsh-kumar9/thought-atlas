"""Re-run the frozen bootstrap analysis and compare every generated output."""
from pathlib import Path
import hashlib,json,datetime
from analyze_multijudge import main
R=Path('/ada1/u/harsh/thought-atlas-reruns/arr_repair_20261101');source=R/'results/multijudge_analysis_indexed';out=R/'results/multijudge_analysis_reproduction_20261105';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['sample_key.parquet','labels_gemma.parquet','labels_qwen.parquet','labels_llama.parquet']:
 p=out/name
 if not p.exists():p.symlink_to(source/name)
main(out)
names=['parse_coverage.csv','pairwise_agreement.csv','trace_profiles.parquet','atlas_weighted_profiles.csv','atlas_contrasts.csv','paired_behavior_shifts.csv','analysis_manifest.json']
checks={n:{'original':sha(source/n),'reproduced':sha(out/n)} for n in names};assert all(v['original']==v['reproduced'] for v in checks.values())
r={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'analysis_source_sha256':sha(R/'scripts/analyze_multijudge.py'),'all_output_hashes_identical':True,'outputs':checks,'scope':'Full frozen deterministic analysis reproduced, including1000-replicate trace-cluster agreement intervals and2000-replicate paired-shift intervals.'};(source/'bootstrap_reproduction_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
