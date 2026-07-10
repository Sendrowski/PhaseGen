"""Build a local, self-contained HTML report from manifest.json: sortable table where clicking a
row opens its diff plot in a lightbox (plots load on demand from sibling PNGs)."""
import json, sys, os

OUTDIR = sys.argv[1]
data = json.load(open(os.path.join(OUTDIR, "manifest.json")))
# empirical = candidate operand is PhaseGen's sampler (a ``...: empirical: ...`` stat path); the render tags each
# row, but fall back to parsing the stat path for manifests written before the flag existed
for r in data:
    r["empirical"] = bool(r.get("empirical", any(s.strip() == "empirical" for s in r["stat"].split(":"))))
scen = sorted(set(r["config"] for r in data))
metrics = sorted(set(r["metric"] for r in data))
n_fail = sum(1 for r in data if r["ratio"] is not None and r["ratio"] > 1)
n_plot = sum(1 for r in data if r["plot"])
n_emp = sum(1 for r in data if r.get("empirical"))
DATA = json.dumps(data, separators=(",", ":"))
META = json.dumps(dict(scenarios=len(scen), comparisons=len(data), metrics=metrics, fails=n_fail,
                       plots=n_plot, empirical=n_emp))

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhaseGen comparison scan</title>
<style>
:root{--bg:#f6f8f9;--surface:#fff;--surface-2:#fbfcfd;--text:#182028;--muted:#5c6a76;--border:#e4e9ed;
 --accent:#0f766e;--accent-soft:#0f766e18;--good:#15803d;--good-bg:#15803d14;--warn:#b45309;--warn-bg:#b4530914;
 --high:#b91c1c;--high-bg:#b91c1c14;--shadow:0 1px 2px #1820280a,0 8px 24px #18202808;}
@media (prefers-color-scheme:dark){:root{--bg:#0d1317;--surface:#141c22;--surface-2:#111820;--text:#e7eef3;
 --muted:#8a97a3;--border:#23303a;--accent:#2dd4bf;--accent-soft:#2dd4bf1f;--good:#4ade80;--good-bg:#4ade8016;
 --warn:#fbbf24;--warn-bg:#fbbf2416;--high:#f87171;--high-bg:#f8717118;--shadow:0 1px 2px #0006,0 10px 30px #0004;}}
:root[data-theme=dark]{--bg:#0d1317;--surface:#141c22;--surface-2:#111820;--text:#e7eef3;--muted:#8a97a3;
 --border:#23303a;--accent:#2dd4bf;--accent-soft:#2dd4bf1f;--good:#4ade80;--good-bg:#4ade8016;--warn:#fbbf24;
 --warn-bg:#fbbf2416;--high:#f87171;--high-bg:#f8717118;--shadow:0 1px 2px #0006,0 10px 30px #0004;}
:root[data-theme=light]{--bg:#f6f8f9;--surface:#fff;--surface-2:#fbfcfd;--text:#182028;--muted:#5c6a76;
 --border:#e4e9ed;--accent:#0f766e;--accent-soft:#0f766e18;--good:#15803d;--good-bg:#15803d14;--warn:#b45309;
 --warn-bg:#b4530914;--high:#b91c1c;--high-bg:#b91c1c14;--shadow:0 1px 2px #1820280a,0 8px 24px #18202808;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1140px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:1.45rem;font-weight:650;letter-spacing:-.01em;margin:0 0 4px;text-wrap:balance}
header p{margin:0;color:var(--muted);font-size:.92rem}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 16px;box-shadow:var(--shadow);min-width:112px}
.tile .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.tile .v{font-size:1.5rem;font-weight:650;margin-top:2px}.tile .v.ok{color:var(--good)}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0 14px}
input[type=search],select{background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:9px;padding:8px 11px;font-size:.9rem;font-family:inherit}
input[type=search]{flex:1;min-width:200px}
input:focus,select:focus,th:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
label.chk{display:flex;align-items:center;gap:6px;font-size:.85rem;color:var(--muted)}
.count{color:var(--muted);font-size:.85rem;margin-left:auto;font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto;border:1px solid var(--border);border-radius:12px;background:var(--surface);box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:.875rem}
thead th{position:sticky;top:0;background:var(--surface-2);text-align:left;padding:11px 14px;font-weight:600;
 font-size:.76rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);border-bottom:1px solid var(--border);
 cursor:pointer;white-space:nowrap;user-select:none}
thead th.num{text-align:right}thead th:hover{color:var(--text)}
th .arrow{opacity:.4;font-size:.7em;margin-left:3px}th[aria-sort] .arrow{opacity:1;color:var(--accent)}
tbody td{padding:9px 14px;border-bottom:1px solid var(--border);white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
tbody tr.has:hover{background:var(--accent-soft);cursor:zoom-in}
tbody tr.sel{background:var(--accent-soft)}
td.num{text-align:right}td.scen{font-weight:500}td.stat{color:var(--muted)}td.metrictag{color:var(--muted);font-size:.8rem}
.pcell{text-align:center;width:34px}
.chip{display:inline-block;padding:1px 8px;border-radius:20px;font-size:.78rem;font-weight:600;font-variant-numeric:tabular-nums}
.chip.good{color:var(--good);background:var(--good-bg)}.chip.warn{color:var(--warn);background:var(--warn-bg)}.chip.high{color:var(--high);background:var(--high-bg)}
footer{margin-top:18px;color:var(--muted);font-size:.8rem}
.modal{position:fixed;inset:0;background:#000a;display:none;align-items:center;justify-content:center;padding:24px;z-index:50}
.modal.open{display:flex}
.mbox{background:var(--surface);border:1px solid var(--border);border-radius:14px;max-width:96vw;max-height:92vh;display:flex;flex-direction:column;box-shadow:0 20px 60px #0007}
.mhead{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--border)}
.mhead .t{font-weight:600;font-size:.95rem}.mhead .m{color:var(--muted);font-size:.82rem;margin-left:auto;font-variant-numeric:tabular-nums}
.mbox img{max-width:min(1100px,94vw);max-height:76vh;object-fit:contain;background:#fff;border-radius:0 0 14px 14px}
.nav{background:none;border:1px solid var(--border);color:var(--text);border-radius:8px;padding:5px 10px;cursor:pointer;font-size:.9rem}
.nav:hover{border-color:var(--accent);color:var(--accent)}
.hint{color:var(--muted);font-size:.78rem}
</style></head><body>
<div class="wrap">
 <header><h1>PhaseGen &harr; msprime comparison scan</h1>
 <p>Every statistic across the non-slow scenario suite. Click a row with a &#128200; to inspect its diff plot; &larr;/&rarr; to step through, Esc to close.</p></header>
 <div class="tiles" id="tiles"></div>
 <div class="controls">
  <input type="search" id="q" placeholder="Filter scenario / statistic / metric&hellip;" aria-label="Filter">
  <select id="metric" aria-label="Metric"></select>
  <label class="chk"><input type="checkbox" id="ponly"> plots only</label>
  <label class="chk"><input type="checkbox" id="eonly"> empirical only</label>
  <span class="count" id="count"></span>
 </div>
 <div class="tablewrap"><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>
 <footer>Sorted by real difference ascending. Raw diffs are not directly comparable across metrics &mdash; use the metric filter or the tolerance-used chip. Plots load on demand from sibling PNG files.</footer>
</div>
<div class="modal" id="modal"><div class="mbox">
 <div class="mhead"><button class="nav" id="prev">&larr;</button><button class="nav" id="next">&rarr;</button>
  <span class="t" id="mt"></span><span class="m" id="mm"></span><button class="nav" id="close">Esc</button></div>
 <img id="mimg" alt="diff plot"></div></div>
<script>
const DATA=__DATA__,META=__META__;
const COLS=[
 {k:'plot',label:'',cls:'pcell',fmt:v=>v?'\u{1F4C8}':''},
 {k:'config',label:'Scenario',cls:'scen'},
 {k:'stat',label:'Statistic',cls:'stat'},
 {k:'diff',label:'Diff',num:true,fmt:fmtNum},
 {k:'tol',label:'Tolerance',num:true,fmt:fmtNum},
 {k:'ratio',label:'Tol. used',num:true,fmt:fmtRatio},
 {k:'metric',label:'Metric',cls:'metrictag'},
 {k:'runtime',label:'Time',num:true,fmt:v=>v.toFixed(3)+'s'}];
function fmtNum(v){if(v==null)return '—';const a=Math.abs(v);return (a!==0&&a<1e-3)?v.toExponential(2):v.toFixed(a<1?5:4);}
function fmtRatio(v){if(v==null)return '—';const c=v<0.6?'good':v<0.85?'warn':'high';return '<span class="chip '+c+'">'+(v*100).toFixed(0)+'%</span>';}
let sortKey='diff',sortDir=1,view=[],cur=-1;
const q=document.getElementById('q'),msel=document.getElementById('metric'),ponly=document.getElementById('ponly'),eonly=document.getElementById('eonly');
const modal=document.getElementById('modal'),mimg=document.getElementById('mimg'),mt=document.getElementById('mt'),mm=document.getElementById('mm');
function tiles(){const t=[['Scenarios',META.scenarios],['Comparisons',META.comparisons.toLocaleString()],
 ['Empirical',(META.empirical||0).toLocaleString()],['With plots',META.plots.toLocaleString()],
 ['Over tolerance',META.fails,META.fails===0?'ok':'']];
 document.getElementById('tiles').innerHTML=t.map(([k,v,c])=>'<div class="tile"><div class="k">'+k+'</div><div class="v '+(c||'')+'">'+v+'</div></div>').join('');}
function head(){document.getElementById('head').innerHTML=COLS.map(c=>{
 const s=c.k===sortKey?(' aria-sort="'+(sortDir>0?'ascending':'descending')+'"'):'';
 const ar=c.k===sortKey?(sortDir>0?'↑':'↓'):(c.label?'↕':'');
 return '<th tabindex="0" data-k="'+c.k+'"'+(c.num?' class="num"':'')+s+'>'+c.label+'<span class="arrow">'+ar+'</span></th>';}).join('');
 document.querySelectorAll('th').forEach(th=>{const f=()=>{const k=th.dataset.k;if(k===sortKey)sortDir*=-1;else{sortKey=k;sortDir=1;}render();};
  th.onclick=f;th.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();f();}};});}
function render(){const term=q.value.trim().toLowerCase(),mf=msel.value,po=ponly.checked,eo=eonly.checked;
 view=DATA.filter(r=>(!mf||r.metric===mf)&&(!po||r.plot)&&(!eo||r.empirical)&&(!term||(r.config+' '+r.stat+' '+r.metric).toLowerCase().includes(term)));
 view.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];if(x==null)x=(typeof y==='number'?Infinity:'');if(y==null)y=(typeof x==='number'?Infinity:'');
  if(typeof x==='string')return sortDir*String(x).localeCompare(String(y));return sortDir*(x-y);});
 document.getElementById('body').innerHTML=view.map((r,i)=>'<tr class="'+(r.plot?'has':'')+'" data-i="'+i+'">'+COLS.map(c=>{
  const raw=r[c.k];const val=c.fmt?c.fmt(raw):(raw==null?'—':raw);
  return '<td class="'+(c.num?'num ':'')+(c.cls||'')+(c.num&&c.k!=='ratio'?' mono':'')+'">'+val+'</td>';}).join('')+'</tr>').join('');
 document.querySelectorAll('tbody tr.has').forEach(tr=>tr.onclick=()=>open(+tr.dataset.i));
 head();document.getElementById('count').textContent=view.length.toLocaleString()+' of '+DATA.length.toLocaleString()+' shown';}
function open(i){const r=view[i];if(!r||!r.plot)return;cur=i;
 mimg.src=r.plot;mt.textContent=r.config+': '+r.stat;
 mm.textContent='diff '+fmtNum(r.diff)+'  ≤  '+fmtNum(r.tol)+'  ('+r.metric+')';
 modal.classList.add('open');
 document.querySelectorAll('tbody tr').forEach(tr=>tr.classList.toggle('sel',+tr.dataset.i===i));}
function step(d){let i=cur;for(let k=0;k<view.length;k++){i=(i+d+view.length)%view.length;if(view[i].plot){open(i);return;}}}
function close(){modal.classList.remove('open');mimg.removeAttribute('src');}
document.getElementById('close').onclick=close;document.getElementById('prev').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);
modal.onclick=e=>{if(e.target===modal)close();};
document.addEventListener('keydown',e=>{if(!modal.classList.contains('open'))return;
 if(e.key==='Escape')close();else if(e.key==='ArrowLeft')step(-1);else if(e.key==='ArrowRight')step(1);});
function metricOpts(){msel.innerHTML='<option value="">All metrics</option>'+META.metrics.map(m=>'<option>'+m+'</option>').join('');}
tiles();metricOpts();render();
q.oninput=render;msel.onchange=render;ponly.onchange=render;eonly.onchange=render;
</script></body></html>"""

HTML = HTML.replace("__DATA__", DATA).replace("__META__", META)
open(os.path.join(OUTDIR, "report.html"), "w").write(HTML)
print("wrote", os.path.join(OUTDIR, "report.html"), "rows:", len(data), "with plots:", n_plot)
