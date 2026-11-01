"""Package only self-contained manuscript sources; never include raw experiment data."""
from pathlib import Path
import hashlib,json,zipfile
P=Path(__file__).resolve().parents[1]
out=P/'thought-atlas-overleaf.zip'
files=[P/'main.tex',P/'references.bib',P/'README.md',P/'acl.sty',P/'acl_natbib.bst',*sorted((P/'sections').glob('*.tex')),*sorted((P/'tables').glob('*.tex')),*sorted((P/'figures').glob('*.pdf')),*sorted((P/'supplement').glob('*.csv')),*sorted((P/'supplement').glob('*.txt'))]
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(P))
manifest=dict(archive=out.name,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),files=[str(p.relative_to(P)) for p in files])
(P/'package_manifest.json').write_text(json.dumps(manifest,indent=2))
print(f'{out}: {len(files)} files; {out.stat().st_size:,} bytes; SHA256 {manifest["sha256"]}')
