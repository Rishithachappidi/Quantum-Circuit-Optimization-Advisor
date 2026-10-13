"""Generate actual Qiskit candidates; preserve the repository's model and rules."""
import ast, json, warnings, os, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import joblib
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
warnings.filterwarnings('ignore')
os.environ['QISKIT_PARALLEL']='FALSE'
ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/'app.py').read_text())
nodes=[]
for node in tree.body:
    if isinstance(node,ast.Import) and all(a.name != 'streamlit' for a in node.names): nodes.append(node)
    elif isinstance(node,ast.ImportFrom): nodes.append(node)
    elif isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) and t.id.isupper() for t in node.targets): nodes.append(node)
    elif isinstance(node,ast.FunctionDef) and node.name not in ['load_local_model','read_dataset_from_path']:
        node.decorator_list=[];nodes.append(node)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_app.py','exec'),ns)
model=joblib.load(ROOT/'random_forest_success_probability.joblib')
original=ns['GENERATORS'].copy()
def make(algorithm,n):
    rng=np.random.default_rng(42)
    if algorithm=='Bell':
        qc=QuantumCircuit(n,n);qc.h(0);qc.cx(0,1);qc.measure(range(n),range(n));return qc
    if algorithm=='Simon':
        pair=original[algorithm](2*(n//2),rng)
        if n%2:
            qc=QuantumCircuit(n,pair.num_clbits);qc.compose(pair,qubits=list(range(n-1)),clbits=list(range(pair.num_clbits)),inplace=True);return qc
        return pair
    return original[algorithm](n,rng)
def packed_ops(qc,limit=180):
    ops=[]
    for inst in qc.data[:limit]:
        ops.append([inst.operation.name,[qc.find_bit(q).index for q in inst.qubits],[float(p) for p in inst.operation.params]])
    return ops
def layers(qc):
    result=[]
    for layer in circuit_to_dag(qc).layers():
        c=Counter(node.op.name for node in layer['graph'].op_nodes())
        # Only names with risk and total count are needed; exact layer counts preserved.
        result.append([c.get('x',0)+c.get('sx',0),c.get('cx',0),c.get('measure',0),sum(c.values())])
    return result
catalog={};start=time.time()
for algorithm in ns['ALGORITHMS']:
    catalog[algorithm]={}
    for n in range(2,11):
        qc=make(algorithm,n);assert qc.num_qubits==n
        for backend in ns['BACKEND_PROFILES']:
            path=ROOT/'docs/data'/f'{algorithm}-{n}-{backend}.json'
            if path.exists():
                catalog[algorithm][f'{n}:{backend}']=path.name;continue
            rows,display,canonical,versions,params=ns['build_candidate_rows'](qc,algorithm,backend,'Medium',1024)
            candidates=[]
            for level,v in versions.items():
                row=rows.iloc[level].to_dict()
                candidates.append({'level':level,'depth':int(row['Optimized_Depth']),'gates':int(row['Optimized_Total_Gates']),'ops':dict(v.count_ops()),'layers':layers(v),'preview':packed_ops(v),'truncated':len(v.data)>180,'totalOperations':len(v.data)})
            predictions={}
            for noise in ns['NOISE_LEVELS']:
                p=ns['hardware_parameters'](backend,noise)
                for shots in ns['SHOT_LEVELS']:
                    x=rows.copy();x['Noise_Model']=noise;x['Shots']=shots;x['Gate_Error_Rate']=p['p2'];x['Readout_Error']=p['readout'];x['Average_T1']=p['t1_us'];x['Average_T2']=p['t2_us']
                    predictions[f'{noise}:{shots}']=np.clip(model.predict(x),0,1).tolist()
            data={'algorithm':algorithm,'qubits':n,'backend':backend,'raw':ns['circuit_features'](canonical),'rawOps':dict(canonical.count_ops()),'rawPreview':packed_ops(canonical),'rawTruncated':len(canonical.data)>180,'candidates':candidates,'predictions':predictions,'seed':42,'qiskit':qiskit.__version__}
            path.write_text(json.dumps(data,separators=(',',':')))
            catalog[algorithm][f'{n}:{backend}']=path.name
        print(f'{algorithm} {n}: ready ({time.time()-start:.1f}s)',flush=True)
    (ROOT/'docs/data/catalog.json').write_text(json.dumps(catalog,separators=(',',':')))
d=pd.read_excel(ROOT/'quantum_circuit_dataset_clean.xlsx')
(ROOT/'docs/data/dataset.json').write_text(d.to_json(orient='records'))
manifest={'source':'https://github.com/Rishithachappidi/Quantum-Circuit-Optimization-Advisor','sourceCommit':'d5d1afc72863ea0c89dad8e28dd076c8e36b82df','qiskit':qiskit.__version__,'seed':42,'profiles':ns['BACKEND_BASE'],'metrics':ns['LATEST_METRICS'],'rows':len(d),'columns':list(d.columns),'originalRanges':ns['ALGORITHM_QUBITS'],'generatedChoices':sum(len(v) for v in catalog.values()),'model':'Original 400-tree Random Forest pipeline; no retraining'}
(ROOT/'docs/data/manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
print('COMPLETE',manifest['generatedChoices'],flush=True)
