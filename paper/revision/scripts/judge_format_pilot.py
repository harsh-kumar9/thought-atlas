"""Diagnostic serialization repair; unchanged prompts/labels, separate outputs.

This does not replace or silently reparse the original frozen pilot. The schema
constrains only the already-requested number/order of sentence indices.
"""
from copy import deepcopy
from pathlib import Path
import argparse,sys

def indexed_schema(schema,n):
    fixed=deepcopy(schema)
    arr=fixed['properties']['sentences'];item=arr.pop('items')
    arr.update(minItems=n,maxItems=n,items=False,prefixItems=[])
    for index in range(1,n+1):
        row=deepcopy(item);row['properties']['index']={'const':index}
        arr['prefixItems'].append(row)
    return fixed

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pilot',type=int,choices=[0,1],default=1)
    a=p.parse_args();sys.path.insert(0,str(a.root))
    from src.judge import run_judge
    base=run_judge.make_sampling
    def constrained(**kwargs):
        n=(kwargs['max_tokens']-128)//64
        assert n>0 and kwargs['max_tokens']==64*n+128
        kwargs['json_schema']=indexed_schema(kwargs['json_schema'],n)
        sampling=base(**kwargs)
        sampling.structured_outputs.whitespace_pattern=r'[ \t]{0,1}'
        return sampling
    run_judge.make_sampling=constrained
    from multijudge import judge
    judge(a.root,a.out.resolve(),'llama',a.pilot,9)
if __name__=='__main__':main()
