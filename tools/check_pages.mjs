import fs from 'node:fs';
import assert from 'node:assert/strict';
import {ALGORITHMS,parameters,exposure,recommend,validateOperations,structure,parseQasm,escapeHTML} from '../docs/engine.js';
const catalog=JSON.parse(fs.readFileSync('docs/data/catalog.json'));
let configurations=0;
for(const algorithm of ALGORITHMS){
  assert.equal(Object.keys(catalog[algorithm]).length,18);
  for(let n=2;n<=10;n++)for(const backend of ['Aer_FullyConnected','Aer_LinearNISQ']){
    const d=JSON.parse(fs.readFileSync('docs/data/'+catalog[algorithm][`${n}:${backend}`]));
    assert.equal(d.qubits,n);assert.equal(d.algorithm,algorithm);assert.equal(d.backend,backend);assert.equal(d.candidates.length,4);
    for(const c of d.candidates){assert.ok(Number.isInteger(c.depth)&&c.depth>=0);assert.ok(Number.isInteger(c.gates)&&c.gates>=0);assert.equal(c.layers.reduce((s,r)=>s+r[3],0),c.totalOperations);assert.ok(c.preview.every(x=>x[1].every(q=>q>=0&&q<n)));}
    for(const noise of ['Ideal','Low','Medium','High'])for(const shots of [256,512,1024,2048,4096]){
      const pred=d.predictions[`${noise}:${shots}`];assert.equal(pred.length,4);assert.ok(pred.every(x=>Number.isFinite(x)&&x>=0&&x<=1));
      const p=parameters(backend,noise),scored=d.candidates.map((c,i)=>({...c,prediction:pred[i],exposure:exposure(c,p).total}));
      assert.ok(scored.every(c=>c.exposure>=0&&c.exposure<=1));if(noise==='Ideal')assert.ok(scored.every(c=>c.exposure===0));
      const winner=recommend(scored);assert.ok(winner.prediction>=Math.max(...pred)-.01);configurations++;
    }
  }
}
assert.equal(configurations,3240);
const bell=structure(2,validateOperations(2,[{gate:'H',q0:0},{gate:'CX',q0:0,q1:1}]));assert.equal(bell.depth,2);assert.equal(bell.gates,2);assert.equal(bell.measure,2);
for(const rows of [[{gate:'CX',q0:0,q1:0}],[{gate:'H',q0:10}],[{gate:'BAD',q0:0}],[{gate:'RZ',q0:0,angle:Infinity}]])assert.throws(()=>validateOperations(10,rows));
const qasm='OPENQASM 2.0; include "qelib1.inc"; qreg q[10]; creg c[10]; h q[0]; rx(pi/2) q[1]; cx q[0],q[1]; measure q -> c;';const parsed=parseQasm(qasm);assert.equal(parsed.n,10);assert.equal(parsed.ops.length,4);assert.equal(parsed.ops[1][2][0],Math.PI/2);
for(const q of ['qreg q[11]; h q[0];','qreg q[2]; cx q[0],q[0];','qreg q[2]; rx(alert(1)) q[0];','qreg q[2]; gate g a { h a; }','qreg q[2]; h q[2];','qreg q[2]; h q[0],q[1];'])assert.throws(()=>parseQasm(q));
assert.equal(escapeHTML('<script>'),'&lt;script&gt;');
const dataset=JSON.parse(fs.readFileSync('docs/data/dataset.json'));assert.equal(dataset.length,1500);assert.equal(Object.keys(dataset[0]).length,37);
console.log('PASS: 162 Qiskit circuit/profile choices; 3,240 noise/shot configurations; custom/QASM validation; original dataset.');
