import sys
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from judge_format_pilot import indexed_schema

def test_exact_indices_and_original_schema_preserved():
    schema={'type':'object','properties':{'sentences':{'type':'array','items':{'type':'object','properties':{'index':{'type':'integer'},'behaviors':{'type':'array','items':{'enum':['check','revise']}}},'required':['index','behaviors'],'additionalProperties':False}}},'required':['sentences']}
    v=Draft202012Validator(indexed_schema(schema,2))
    assert schema['properties']['sentences']['items']['properties']['index']=={'type':'integer'}
    valid={'sentences':[{'index':1,'behaviors':['check']},{'index':2,'behaviors':[]}]}
    v.validate(valid)
    for rows in [valid['sentences'][:1],valid['sentences']+valid['sentences'][:1],[valid['sentences'][0]]*2,list(reversed(valid['sentences']))]:
        assert not v.is_valid({'sentences':rows})
