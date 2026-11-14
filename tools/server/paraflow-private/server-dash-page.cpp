// ParaFlow built-in dashboard page (served at /paraflow/dash when started with --dash).
// Self-contained: no external scripts, fonts or CDNs. /paraflow/dash/snap returns the latest counter snapshot and,
// with ?h=<bucket>, the server's 6 h throughput history (2 s buckets); the page only aggregates and draws it.

const char * paraflow_dash_html();

const char * paraflow_dash_html() {
    return R"PFDASH(<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ParaFlow 中控</title>
<style>
:root{--page:#f9f9f7;--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e0;--dec:#2a78d6;--pf:#eb6834;--aq:#1baf7a;--yl:#eda100;--mg:#e87ba4;--vi:#6250d6;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--idle:#dcdbd5}
@media (prefers-color-scheme:dark){:root{--page:#0d0d0d;--surf:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--dec:#3987e5;--pf:#d95926;--aq:#199e70;--yl:#c98500;--mg:#d55181;--vi:#9085e9;--idle:#383835}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:13px/1.4 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
header{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline;padding:10px 16px 4px}
h1{font-size:16px;margin:0 8px 0 0}.chip{color:var(--ink2);white-space:nowrap}.chip b{color:var(--ink);font-weight:600}
main{display:grid;grid-template-columns:repeat(12,1fr);gap:12px;padding:8px 16px 16px}
.card{background:var(--surf);border:1px solid var(--grid);border-radius:8px;padding:10px 12px;min-width:0}
.card h2{font-size:13px;margin:0 0 8px;color:var(--ink2);font-weight:600;display:flex;justify-content:space-between;gap:8px}
.w12{grid-column:span 12}.w8{grid-column:span 8}.w6{grid-column:span 6}.w4{grid-column:span 4}
@media (max-width:1100px){.w8,.w6,.w4{grid-column:span 12}}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:8px}
.tile{padding:6px 8px;border-radius:6px;border:1px solid var(--grid)}.tile .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}.tile .k{color:var(--ink2);font-size:12px}
.tile.bad{border-color:var(--crit)}.tile.bad .v{color:var(--crit)}
canvas{width:100%;display:block}
.seg button{background:none;border:1px solid var(--grid);color:var(--ink2);padding:1px 8px;cursor:pointer;font:inherit}
.seg button.on{background:var(--ink);color:var(--surf);border-color:var(--ink)}.seg button:first-child{border-radius:4px 0 0 4px}.seg button:last-child{border-radius:0 4px 4px 0}
.legend{display:flex;gap:14px;color:var(--ink2);font-size:12px;flex-wrap:wrap}.sw{display:inline-block;width:14px;height:10px;margin-right:4px;vertical-align:-1px;border-radius:2px}
.kv{display:grid;grid-template-columns:auto 1fr auto;gap:4px 10px;align-items:center;font-variant-numeric:tabular-nums}
.bar{height:12px;background:var(--grid);border-radius:3px;overflow:hidden;display:flex}.bar i{display:block;height:100%}
.muted{color:var(--ink2)}.num{font-variant-numeric:tabular-nums;text-align:right}
.lanes{display:flex;flex-direction:column;gap:4px}.lane{display:flex;gap:3px;align-items:center;flex-wrap:wrap}.lane .lab{width:52px;color:var(--ink2);font-size:12px}
.s{width:38px;height:24px;border-radius:3px;font-size:10px;display:flex;align-items:center;justify-content:center;color:#fff;position:relative;background:var(--idle)}
.s.idle{color:var(--ink2)}.s.pf{background:var(--pf)}.s.gen{background:var(--dec)}.s.pause{background:repeating-linear-gradient(45deg,var(--idle),var(--idle) 3px,var(--ink2) 3px,var(--ink2) 5px)}
.s.out{background:none;border:1.5px dashed var(--ink2);color:var(--ink)}.s.vip{outline:2px solid var(--mg);outline-offset:-1px}.s .m{position:absolute;right:2px;top:2px;width:5px;height:5px;border-radius:50%;background:#fff}
table{border-collapse:collapse;width:100%}td,th{padding:2px 6px;text-align:left}th{color:var(--ink2);font-weight:500}td.n{text-align:right;font-variant-numeric:tabular-nums}
#err{color:var(--crit);padding:0 16px}.note{font-size:11px;color:var(--ink2);margin-top:6px}
#tip{position:fixed;pointer-events:none;background:var(--surf);border:1px solid var(--grid);border-radius:4px;padding:4px 8px;font-size:12px;display:none;font-variant-numeric:tabular-nums;z-index:9}
</style></head><body>
<header><h1 data-en="ParaFlow Console">ParaFlow 中控</h1><span id="cfg" style="display:flex;flex-wrap:wrap;gap:2px 12px;flex:1 1 400px"></span><select id="lang" aria-label="语言 / Language" style="margin-left:auto;align-self:center;font:inherit;color:var(--ink);background:var(--surf);border:1px solid var(--grid);border-radius:4px;padding:1px 4px"><option value="zh">中文</option><option value="en">English</option></select></header>
<div id="err"></div>
<main>
 <section class="card w12"><div class="tiles" id="tiles"></div></section>
 <section class="card w12"><h2><span data-en="Throughput">吞吐</span>
  <span class="seg" id="win"><button data-s="60" data-en="1m">1分</button><button data-s="600" class="on" data-en="10m">10分</button><button data-s="3600" data-en="1h">1时</button><button data-s="21600" data-en="6h">6时</button></span></h2>
  <div class="legend"><b>decode</b><span><i class="sw" style="background:var(--dec);opacity:.45"></i><span data-en="output t/s (fill, left axis)">输出 t/s（填充，左轴）</span></span><span><i class="sw" style="height:0;border-radius:0;border-top:2px solid var(--ink);vertical-align:3px"></i><span data-en="requests generating (line, right axis; paused excluded)">正在生成的请求数（实线，右轴；不含暂停）</span></span><span><i class="sw" style="background:var(--aq)"></i><span data-en="bottom bar: pure-decode periods">底部条：纯 decode 时段</span></span></div>
  <canvas id="tpd" data-h="190"></canvas>
  <div class="legend" style="margin-top:10px"><b>prefill</b><span><i class="sw" style="background:var(--pf);opacity:.45"></i><span data-en="compute t/s (fill, left axis)">计算 t/s（填充，左轴）</span></span><span><i class="sw" style="height:0;border-radius:0;border-top:2px solid var(--ink);vertical-align:3px"></i><span data-en="requests in prefill (line, right axis; paused excluded)">正在 prefill 的请求数（实线，右轴；不含暂停）</span></span></div>
  <canvas id="tpp" data-h="170"></canvas>
  <div class="legend" style="margin-top:4px"><span id="winstat"></span></div>
 </section>
 <section class="card w6"><h2><span data-en="MTP drafts">MTP 草稿</span><span class="muted" id="mwin"></span></h2><div id="mtp"></div></section>
 <section class="card w6"><h2><span data-en="VRAM (allocator view)">显存（分配器口径）</span><span class="muted" id="memage"></span></h2><div id="mem"></div>
  <div class="note" data-en="Counted by the ggml allocator: weights / KV+state / compute buffers. Excludes CUDA context and driver overhead, so it reads lower than nvidia-smi.">按 ggml 分配器统计：权重 / KV+状态 / 计算缓冲。不含 CUDA 上下文与驱动开销，比 nvidia-smi 低。</div></section>
 <section class="card w6"><h2><span data-en="KV and system cache">KV 与系统缓存</span></h2><div id="kv"></div></section>
 <section class="card w6"><h2><span data-en="Requests swapped to RAM">换出到内存的请求</span><span class="muted" id="suspsum"></span></h2><div id="susp"></div>
  <div class="note" data-en="When a VIP request arrives, ordinary requests are first paused in place (KV stays in VRAM). Only when no slot is free is a paused request swapped out to RAM whole; the VIP borrows its slot, and the request is swapped back afterwards. Requests are also swapped out when the KV pool is full.">VIP 到来时先就地暂停普通请求（KV 留在显存）；没有空槽时才把暂停的请求整体换出到内存，VIP 借用它的槽位，结束后再换回。KV 池不够时也会换出（“容量”）。</div></section>
 <section class="card w12"><h2><span data-en="Context length at request completion">请求完成时上下文长度分布</span><span class="muted" id="chstat"></span><label style="margin-left:auto;display:flex;align-items:center;gap:6px;font-weight:400;font-size:13px"><span data-en="Bin width">分组宽度</span><input type="range" id="chbw" min="0" max="3" value="0" style="width:80px"><span id="chbwval">1k</span></label></h2><canvas id="ch" height="160"></canvas></section>
 <section class="card w12"><h2><span data-en="Request slots (by lane)">请求槽位（按 lane）</span><span class="legend"><span><i class="sw" style="background:var(--dec)"></i><span data-en="Generating">生成</span></span><span><i class="sw" style="background:var(--pf)"></i>prefill</span><span><i class="sw" style="background:var(--idle)"></i><span data-en="Idle">空闲</span></span><span><i class="sw" style="background:var(--idle);outline:1px solid var(--ink2)"></i><span data-en="Paused">暂停</span></span><span><i class="sw" style="outline:2px solid var(--mg)"></i>VIP</span><span>● MTP</span><span><i class="sw" style="border:1.5px dashed var(--ink2)"></i><span data-en="Swapped to RAM">已换出到内存</span></span></span></h2><div class="lanes" id="lanes"></div>
  <div class="note" data-en="The slot count is fixed by the -np startup flag. VIP adds no slots: it pauses ordinary requests or borrows the slot of a swapped-out request.">槽位数固定为启动参数 -np。VIP 不增加槽位：它暂停普通请求或借用被换出请求的槽位。</div></section>
 <section class="card w12"><h2><span data-en="Errors (since start / new in window)">异常计数（启动以来 / 窗口内新增）</span></h2><div id="events"></div></section>
 <section class="card w12"><details><summary class="muted" style="cursor:pointer"><span data-en="Engine diagnostics (for developers; safe to ignore)">引擎诊断（开发用，日常可忽略）</span><span id="pwin" style="margin-left:8px"></span></summary><div id="pipe" style="margin-top:8px"></div></details></section>
</main>
<div id="tip"></div>
<script>
"use strict";
const $=id=>document.getElementById(id);
const KEY='pfdash:'+location.host;
// UI language: zh / en. ?lang=en|zh in the URL wins, then the header dropdown choice kept in localStorage, then the browser language.
let LANG=new URLSearchParams(location.search).get('lang')||localStorage.getItem('pfdash:lang')||((navigator.language||'').toLowerCase().startsWith('zh')?'zh':'en');if(LANG!=='en')LANG='zh';
const tr=(zh,en)=>LANG==='en'?en:zh;
function applyLang(){document.documentElement.lang=LANG==='en'?'en':'zh';document.title=tr('ParaFlow 中控','ParaFlow Console');$('lang').value=LANG;
  document.querySelectorAll('[data-en]').forEach(el=>{if(el.dataset.zh==null)el.dataset.zh=el.textContent;el.textContent=LANG==='en'?el.dataset.en:el.dataset.zh})}
$('lang').onchange=e=>{LANG=e.target.value==='en'?'en':'zh';try{localStorage.setItem('pfdash:lang',LANG)}catch(x){}applyLang();if(last)render(last);draw()};
applyLang();
let apiKey=localStorage.getItem(KEY+':key')||'';
let win=600, last=null, cfg=null;
// Hist: server throughput buckets by absolute index (wall ms / BMS). A row is [dp, dd, dd_pure, ng_ms, np_ms, pure_ms, mix_ms]
// (tokens and request-milliseconds), 0 for an idle bucket; indices before this server process started are absent.
// The open bucket is never sent; each poll re-asks for the last few complete ones, so late spreads are picked up.
let Hist=new Map(),BMS=2000,hStart=0,hCur=0,tStart=null;
document.querySelectorAll('#win button').forEach(b=>b.onclick=()=>{win=+b.dataset.s;document.querySelectorAll('#win button').forEach(x=>x.classList.toggle('on',x===b));draw()});
const f0=x=>x==null||!isFinite(x)?'–':Math.round(x).toLocaleString();
const f1=x=>x==null||!isFinite(x)?'–':x.toFixed(1);
const pct=x=>x==null||!isFinite(x)?'–':(x*100).toFixed(1)+'%';
const gib=b=>(b/1073741824).toFixed(2);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
// history of cumulative snapshots to compute window deltas for the side panels
let H=[];
async function poll(){
  let d;
  try{
    const from=hCur&&Hist.size?hCur-5:0;
    const r=await fetch(location.pathname.replace(/\/?$/,'/snap')+'?h='+from,{headers:apiKey?{'Authorization':'Bearer '+apiKey}:{},cache:'no-store'});
    if(r.status===401){const k=prompt(tr('需要 API key','API key required'));if(k){apiKey=k;localStorage.setItem(KEY+':key',k)}return}
    d=await r.json();$('err').textContent='';
  }catch(e){$('err').textContent=tr('连接失败：','Connection failed: ')+e;return}
  if(!d.ok){$('err').textContent=tr('等待首个快照…','Waiting for the first snapshot…');return}
  if(d.config)cfg=d.config;
  const w=Date.now();
  if(last&&(d.t_start_us!==last.t_start_us||d.predict<last.predict||d.prompt<last.prompt))H=[]; // server restarted: window deltas start over
  if(d.hist){const h=d.hist;
    // another server process: forget the old one's buckets and fetch the new one's full history on the next poll
    if(tStart!==null&&tStart!==d.t_start_us){Hist=new Map();hCur=0;tStart=d.t_start_us;setTimeout(poll,0);return}tStart=d.t_start_us;
    BMS=h.bucket_ms;hStart=h.start;hCur=h.cur;
    h.rows.forEach((v,k)=>Hist.set(h.from+k,v));
    const lo=hCur-6*3600e3/BMS;for(const k of Hist.keys()){if(k<lo)Hist.delete(k);else break}}
  d._w=w;last=d;{const h={};for(const k in d){if(typeof d[k]==='number')h[k]=d[k]}h.accepted_pos=d.accepted_pos;h.ctx_hist=d.ctx_hist;H.push(h)}while(H.length&&w-H[0]._w>win*1000+2000)H.shift();
  render(d);draw();
}
function base(){ // oldest snapshot inside the window
  return H.length?H[0]:null;
}
function render(d){
  if(cfg){const c=cfg;$('cfg').innerHTML=[[tr('模型','Model'),c.model],['C',c.n_parallel],[tr('上下文池','Context pool'),f0(c.n_ctx)],[tr('单路','Per slot'),f0(c.n_ctx_slot)],['ub',c.n_ubatch],['b',c.n_batch],['lanes',c.pipe_lanes],['prefill lanes',c.prefill_lanes],['chunk',c.prefill_chunk],['KV',c.cache_type_k+'/'+c.cache_type_v],['MTP',c.mtp?('n_max '+c.draft_n_max):tr('关','off')],[tr('系统缓存','System cache'),c.system_cache_mib+' MiB'],['VIP',c.vip?tr('开','on'):tr('关','off')],[tr('卡','GPUs'),(c.devices||[]).length],[tr('构建','Build'),c.build]].map(([k,v])=>`<span class="chip">${k} <b>${v}</b></span>`).join('')}
  // instantaneous rates from the last ~5 samples
  let dt=0,dp=0,dd=0;for(let k=hCur-1;k>=hCur-3&&k>=hStart;k--){const v=Hist.get(k);if(v===undefined)break;dt+=BMS/1e3;if(v){dp+=v[0];dd+=v[1]}}
  const b=base(),span=b?(d.t_us-b.t_us)/1e6:0,D=k=>b?d[k]-b[k]:0;
  const acc=D('draft_tokens')>0?D('draft_accepted')/D('draft_tokens'):null;
  const errs=d.oom_isolations+d.oom_terminated+d.draft_failures+d.sc_failures;
  const tiles=[[tr('运行中','Running'),d.n_processing+' / '+d.n_slots],[tr('生成','Generating'),d.n_gen],['prefill',d.n_prompt],[tr('暂停','Paused'),d.n_paused,d.n_paused>0?'':null],[tr('换出到内存','Swapped to RAM'),d.n_suspended,d.n_suspended>0?'':null],['VIP',d.n_vip],[tr('排队','Queued'),d.n_deferred],
    ['decode t/s (6s)',f0(dt?dd/dt:null)],['prefill t/s (6s)',f0(dt?dp/dt:null)],[tr('MTP 接受率','MTP acceptance'),pct(acc)],[tr('KV 占用','KV used'),pct(d.kv_cap?d.kv_used/d.kv_cap:null)],[tr('异常','Errors'),errs,errs>0?'bad':'']];
  $('tiles').innerHTML=tiles.map(([k,v,c])=>`<div class="tile ${c||''}"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');
  // pipeline
  $('pwin').textContent=span?Math.round(span)+tr(' 秒窗口',' s window'):'';
  const cars=D('cars_prefill')+D('cars_decode')+D('cars_mixed');
  const wall=span*1e6||1;
  const rows=[[tr('主线程提交计算','Main thread: submit compute'),D('issue_us')/wall],[tr('主线程等 GPU 结果','Main thread: wait for GPU'),D('wait_us')/wall],[tr('主线程处理结果（采样/发送）','Main thread: process results (sample/send)'),D('process_us')/wall],[tr('等流水线清空','Wait for pipeline drain'),D('drain_us')/wall],[tr('新请求接入时清空','Drain on new-request admission'),D('admission_us')/wall],[tr('等 MTP 草稿线程','Wait for MTP draft thread'),D('dft_join_wait_us')/wall]];
  $('pipe').innerHTML=`<div class="kv">
   <span>${tr('批次/秒','Batches/s')}</span><div class="bar">${cars?[['cars_prefill','--pf'],['cars_mixed','--yl'],['cars_decode','--dec']].map(([k,c])=>`<i style="width:${D(k)/cars*100}%;background:var(${c})" title="${k}"></i>`).join(''):''}</div><span class="num">${f0(cars/(span||1))}/s</span>
   <span class="muted" style="grid-column:1/4">${tr('发往 GPU 的批次：纯 prefill ','Batches sent to GPU: pure prefill ')}${f0(D('cars_prefill'))} · ${tr('混合 ','mixed ')}${f0(D('cars_mixed'))} · ${tr('纯 decode ','pure decode ')}${f0(D('cars_decode'))}${tr('；平均每批 prompt ','; avg rows per batch: prompt ')}${f1(D('payload_prompt')/Math.max(1,D('cars_prefill')+D('cars_mixed')))}${tr(' 行 · decode ',' · decode ')}${f1(D('payload_decode')/Math.max(1,D('cars_decode')+D('cars_mixed')))}${tr(' 行','')}</span>
   ${rows.map(([k,v])=>`<span>${k}</span><div class="bar"><i style="width:${Math.min(100,v*100)}%;background:var(--vi)"></i></div><span class="num">${pct(v)}</span>`).join('')}
  </div><div class="note">${tr('占比 = 主线程在该环节的累计时间 / 墙钟时间；“等 GPU”高说明 GPU 是瓶颈，“处理结果”高说明 CPU 侧是瓶颈。KV 整理移动 ','Share = main-thread time spent in the step / wall time. High “wait for GPU” means the GPU is the bottleneck; high “process results” means the CPU side is. KV compaction moves ')}${f0(D('compact_moves'))} · ${tr('admission 跳过 ','admission skipped ')}${f0(D('admission_skipped'))} · ${tr('VIP 跳过 ','VIP skipped ')}${f0(D('vip_pause_skipped'))} · ${tr('checkpoint 保留 ','checkpoint holds ')}${f0(D('ckpt_holds'))} · ${tr('延后 ','deferrals ')}${f0(D('deferrals'))}</div>`;
  // mtp
  $('mwin').textContent=span?Math.round(span)+tr(' 秒窗口',' s window'):'';
  const steps=D('draft_steps');
  const pos=(d.accepted_pos||[]).map((v,i)=>steps>0?(v-((b&&b.accepted_pos&&b.accepted_pos[i])||0))/steps:null);
  $('mtp').innerHTML=!cfg||!cfg.mtp?'<span class="muted">'+tr('未启用 MTP','MTP disabled')+'</span>':`<div class="kv">
   <span>${tr('接受率','Acceptance')}</span><div class="bar"><i style="width:${(acc||0)*100}%;background:var(--aq)"></i></div><span class="num">${pct(acc)}</span>
   <span>${tr('平均接受长度','Mean accepted length')}</span><span></span><span class="num">${steps>0?f1(1+D('draft_accepted')/steps):'–'}</span>
   ${pos.map((v,i)=>`<span class="muted">${tr('位置 ','Position ')}${i+1}</span><div class="bar"><i style="width:${(v||0)*100}%;background:var(--aq)"></i></div><span class="num">${pct(v)}</span>`).join('')}
   <span>${tr('草稿 join','Draft joins')}</span><span class="muted">${f0(D('dft_join_waits'))} / ${f0(D('dft_joins'))}${tr(' 次需等待',' had to wait')}</span><span class="num">${f1(D('dft_join_wait_us')/Math.max(1,D('dft_join_waits'))/1000)} ms</span>
  </div><div class="note">${tr('接受率含进行中请求。降级 ','Acceptance includes in-flight requests. Degrades ')}${d.draft_degrade} · ${tr('草稿步失败 ','draft step failures ')}${d.draft_failures}</div>`;
  // memory
  $('memage').textContent=d.mem_t_us?`${f1((d.t_us-d.mem_t_us)/1e6)}`+tr(' 秒前',' s ago'):'';
  const tot=new Map(((cfg&&cfg.devices)||[]).map(x=>[x.name,x.total]));
  $('mem').innerHTML='<div class="kv">'+(d.devices||[]).map(e=>{const t=e.total||tot.get(e.name)||(e.model+e.context+e.compute),u=e.model+e.context+e.compute;
    return `<span>${e.name}</span><div class="bar" title="${tr('权重','weights')} ${gib(e.model)} / KV ${gib(e.context)} / ${tr('计算','compute')} ${gib(e.compute)} GiB"><i style="width:${e.model/t*100}%;background:var(--ink2)"></i><i style="width:${e.context/t*100}%;background:var(--dec)"></i><i style="width:${e.compute/t*100}%;background:var(--yl)"></i></div><span class="num">${gib(u)} / ${gib(t)} GiB</span>`}).join('')+
    `</div><div class="legend" style="margin-top:4px"><span><i class="sw" style="background:var(--ink2)"></i>${tr('权重','Weights')}</span><span><i class="sw" style="background:var(--dec)"></i>${tr('KV / 状态','KV / state')}</span><span><i class="sw" style="background:var(--yl)"></i>${tr('计算缓冲','Compute buffers')}</span></div>`;
  // kv + system cache
  const look=D('sc_hits')+D('sc_misses');
  $('kv').innerHTML=`<div class="kv">
   <span>${tr('KV 池','KV pool')}</span><div class="bar" style="position:relative"><i style="width:${d.kv_cap?d.kv_used/d.kv_cap*100:0}%;background:var(--dec)"></i>${d.kv_max&&d.kv_cap?`<i style="position:absolute;top:0;bottom:0;left:${Math.min(99.5,d.kv_max/d.kv_cap*100)}%;width:2px;background:#e53;border-radius:0;padding:0;margin:0"></i>`:''}</div><span class="num">${f0(d.kv_used)} / ${f0(d.kv_cap)}${d.kv_max?tr(' 峰 ',' peak ')+f0(d.kv_max):''}</span>
   <span>${tr('系统缓存命中','System cache hit rate')}</span><div class="bar"><i style="width:${look?D('sc_hits')/look*100:0}%;background:var(--aq)"></i></div><span class="num">${look?pct(D('sc_hits')/look):'–'}</span>
   <span>${tr('缓存占用','Cache used')}</span><div class="bar"><i style="width:${cfg&&cfg.system_cache_mib?d.sc_bytes/(cfg.system_cache_mib*1048576)*100:0}%;background:var(--aq)"></i></div><span class="num">${f0(d.sc_bytes/1048576)} MiB</span>
   <span>${tr('窗口内命中 token','Hit tokens in window')}</span><span></span><span class="num">${f0(D('sc_hit_tokens'))}</span>
   <span>${tr('prompt 缓存复用','Prompt cache reuse')}</span><span></span><span class="num">${f0(D('cached'))}</span>
  </div>`;
  // lanes
  const by=new Map();for(const s of d.slots||[]){const[id,st,lane,stream,tok,cells,vip,mtp,paused]=s;if(!by.has(lane))by.set(lane,[]);by.get(lane).push(s)}
  $('lanes').innerHTML=[...by.keys()].sort((a,b)=>a-b).map(l=>`<div class="lane"><span class="lab">lane ${l}</span>${by.get(l).map(([id,st,lane,stream,tok,cells,vip,mtp,paused])=>{
     const cls=paused?'pause':st===5?'gen':st>=2?'pf':'idle';
     return `<div class="s ${cls}${vip?' vip':''}" title="slot ${id} · stream ${stream} · ${tok} tok · KV ${cells}">${cells?(cells>=1000?(cells/1000).toFixed(cells>=10000?0:1)+'k':cells):id}${mtp?'<i class="m"></i>':''}</div>`}).join('')}</div>`).join('');
  // paged-out requests
  const sp=d.susp||[];
  $('lanes').innerHTML+=sp.length?`<div class="lane"><span class="lab">${tr('内存','RAM')}</span>${sp.map(([id,tok,vip,cap,by])=>`<div class="s out${vip?' vip':''}" title="${tr('原 slot ','from slot ')}${id} · ${tok} tok · ${(by/1048576).toFixed(0)} MiB · ${cap?tr('KV 容量不足换出','swapped out: KV pool full'):tr('为 VIP 让出槽位','yielded slot to VIP')}">${tok>=1000?(tok/1000).toFixed(tok>=10000?0:1)+'k':tok}</div>`).join('')}</div>`:'';
  $('suspsum').textContent=sp.length?`${sp.length}`+tr(' 个',' requests')+` · ${gib(d.susp_bytes||0)} GiB`:'';
  $('susp').innerHTML=sp.length?`<table><tr><th>${tr('原槽位','Slot')}</th><th>token</th><th>${tr('类型','Type')}</th><th>${tr('原因','Reason')}</th><th class="num">${tr('内存','RAM')}</th></tr>`+sp.map(([id,tok,vip,cap,by])=>`<tr><td>slot ${id}</td><td class="n">${f0(tok)}</td><td>${vip?'VIP':tr('普通','ordinary')}</td><td>${cap?tr('KV 容量不足','KV pool full'):tr('给 VIP 让槽位','yielded to VIP')}</td><td class="n">${(by/1048576).toFixed(0)} MiB</td></tr>`).join('')+'</table>':'<span class="muted">'+tr('当前没有换出的请求','No requests swapped out')+'</span>';
  // events
  const ev=[[tr('OOM 隔离','OOM isolations'),'oom_isolations'],[tr('OOM 终止','OOM terminations'),'oom_terminated'],[tr('草稿降级','Draft degrades'),'draft_degrade'],[tr('草稿步失败','Draft step failures'),'draft_failures'],[tr('系统缓存恢复失败','System cache restore failures'),'sc_failures'],[tr('系统缓存淘汰','System cache evictions'),'sc_evictions']];
  $('events').innerHTML='<table><tr>'+ev.map(([k])=>`<th>${k}</th>`).join('')+'</tr><tr>'+ev.map(([k,f])=>{const n=D(f);return `<td class="n" style="${(f!=='sc_evictions'&&d[f]>0)?'color:var(--crit)':''}">${f0(d[f])} <span class="muted">${n?'+'+n:''}</span></td>`}).join('')+'</tr></table>';
  drawCtxHist(d);
}
function draw(){
  // time axis = server wall clock; hCur*BMS is the start of the open (not yet sent) bucket, so curves end there
  const now=hCur?hCur*BMS:Date.now(),t0=now-win*1000;
  // display bins are whole multiples of the server bucket, aligned to the wall-clock grid
  const per=Math.max(1,Math.round(win*1000/150/BMS)),bw=per*BMS,origin=Math.floor(t0/bw)*bw,nb=Math.ceil((now-origin)/bw)+1;
  const bins=Array.from({length:nb},()=>({dt:0,dp:0,dd:0,pure:0,gap:false,ng:0,npv:0,wt:0}));
  let pureT=0,pureD=0,mixT=0,mixD=0,allT=0,allD=0,allP=0;
  for(let k=Math.floor(origin/BMS);k<hCur;k++){const ts=k*BMS;if(ts+BMS<=t0)continue;
    const b=bins[Math.floor((ts-origin)/bw)];if(!b)continue;const v=Hist.get(k);
    if(v===undefined)continue; // before this server process started: no data, drawn empty
    const sec=BMS/1e3;b.dt+=sec;b.wt+=BMS;allT+=sec;if(!v)continue;
    b.dp+=v[0];b.dd+=v[1];b.ng+=v[3];b.npv+=v[4];b.pure+=v[5]/1e3;allD+=v[1];allP+=v[0];
    pureT+=v[5]/1e3;pureD+=v[2];mixT+=v[6]/1e3;mixD+=v[1]-v[2]}
  // red line where the current server process started (its history begins there)
  {const ts=hStart*BMS;if(hStart&&ts>t0){const b=bins[Math.floor((ts-origin)/bw)];if(b)b.mark=true}}
  const ok=b=>b.dt>0&&!b.gap;
  // 3-point smoothing of displayed rates (within continuous, non-gap runs)
  const rawD=bins.map(b=>ok(b)?b.dd/b.dt:NaN),rawP=bins.map(b=>ok(b)?b.dp/b.dt:NaN);
  const smooth=raw=>raw.map((v,i)=>{if(isNaN(v))return NaN;let s=0,n=0;for(let j=Math.max(0,i-1);j<=Math.min(raw.length-1,i+1);j++){if(!isNaN(raw[j])&&!bins[j].gap){s+=raw[j];n++}}return n?s/n:v});
  const smD=smooth(rawD),smP=smooth(rawP);
  // per-bin request counts: time-weighted mean of generating / prefilling requests (paused excluded), from the server
  const cntNG=bins.map(b=>(b.wt>0&&!b.gap)?b.ng/b.wt:NaN),cntNP=bins.map(b=>(b.wt>0&&!b.gap)?b.npv/b.wt:NaN);
  const runs=(arr,X)=>{const out=[];let cur=null;for(let i=0;i<nb-1;i++){const bs=origin+i*bw,be=origin+(i+1)*bw;if(be<=t0||bs>=now||isNaN(arr[i])||bins[i].gap){cur=null;continue}if(!cur){cur=[];out.push(cur)}cur.push([X(i),arr[i]])}return out};
  // one chart: rate as a filled area on the left axis, request count as an integer staircase line on the right axis
  const panel=(cv,o)=>{
    const dpr=devicePixelRatio||1,W=cv.clientWidth,Hh=+cv.dataset.h;cv.width=W*dpr;cv.height=Hh*dpr;cv.style.height=Hh+'px';
    const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,W,Hh);
    const L=72,R=50,T=8,B=o.pure?30:22,pw=W-L-R,ph=Hh-T-B,ink=css('--ink');
    let ym=10;for(const v of o.rate)if(!isNaN(v))ym=Math.max(ym,v);const sR=niceStep(ym/4);ym=Math.ceil(ym/sR)*sR;
    let cm=o.cmax||4;if(!o.cmax)for(const v of o.cnt)if(!isNaN(v))cm=Math.max(cm,Math.round(v));const sN=niceStep(Math.max(1,cm/4));cm=Math.ceil(cm/sN)*sN;
    const X=i=>L+(origin+(i+.5)*bw-t0)/(win*1000)*pw,Y=v=>T+ph-v/ym*ph,YN=v=>T+ph-v/cm*ph;
    g.font='11px system-ui,sans-serif';g.strokeStyle=css('--grid');g.lineWidth=1;
    for(let v=0;v<=ym+1e-9;v+=sR){const y=Math.round(Y(v))+.5;g.beginPath();g.moveTo(L,y);g.lineTo(L+pw,y);g.stroke();g.fillStyle=o.col;g.textAlign='right';g.fillText(f0(v),L-6,y+4)}
    g.fillStyle=ink;g.textAlign='left';for(let v=0;v<=cm+1e-9;v+=sN)g.fillText(f0(v),L+pw+6,YN(v)+4);
    g.fillStyle=css('--ink2');g.textAlign='center';for(let k=0;k<=4;k++){const t=t0+k*win*250;g.fillText(new Date(t).toTimeString().slice(0,win>=3600?5:8),L+k/4*pw,Hh-B+(o.pure?20:15))}
    g.save();g.translate(11,T+ph/2);g.rotate(-Math.PI/2);g.textAlign='center';g.fillStyle=o.col;g.fillText(o.name+' t/s',0,0);g.restore();
    g.save();g.translate(W-8,T+ph/2);g.rotate(Math.PI/2);g.fillStyle=ink;g.fillText(o.cl,0,0);g.restore();
    for(const r of runs(o.rate,X)){g.beginPath();g.moveTo(r[0][0],Y(0));for(const[x,v]of r)g.lineTo(x,Y(v));g.lineTo(r[r.length-1][0],Y(0));g.closePath();g.fillStyle=o.col;g.globalAlpha=.45;g.fill();g.globalAlpha=1}
    const bwpx=bw/(win*1000)*pw;g.strokeStyle=ink;g.lineWidth=1.5;g.lineJoin='miter';
    for(const r of runs(o.cnt,X)){g.beginPath();r.forEach(([x,v],k)=>{const y=Math.round(YN(Math.round(v)))+.5;if(k)g.lineTo(x-bwpx*.5,y);else g.moveTo(x-bwpx*.5,y);g.lineTo(x+bwpx*.5,y)});g.stroke()}
    for(let i=0;i<nb;i++){const bs=origin+i*bw,be=origin+(i+1)*bw;if(be<=t0||bs>=now)continue;
      const x0=L+Math.max(0,(bs-t0)/(win*1000))*pw,x1=L+Math.min(1,(be-t0)/(win*1000))*pw,b=bins[i];
      if(o.pure&&b.dt>0&&b.pure>0){g.fillStyle=css('--aq');g.globalAlpha=Math.min(1,.25+.75*b.pure/b.dt);g.fillRect(x0,T+ph+3,Math.max(1,x1-x0),4);g.globalAlpha=1}
      if(b.mark){g.strokeStyle=css('--crit');g.lineWidth=1.5;g.beginPath();g.moveTo(x0,T);g.lineTo(x0,T+ph);g.stroke()}}
    cv._bins={bins,nb,L,pw,t0,bw,origin,rate:o.rate,cnt:o.cnt,name:o.name,cl:o.cl,pure:o.pure};
  };
  // decode count axis = -np (shows how close to full); prefill count axis = observed max (usually a few)
  panel($('tpd'),{name:'decode',rate:smD,cnt:cntNG,col:css('--dec'),cl:tr('生成中请求数','requests generating'),cmax:cfg?Math.max(1,cfg.n_parallel):0,pure:true});
  panel($('tpp'),{name:'prefill',rate:smP,cnt:cntNP,col:css('--pf'),cl:tr('prefill 中请求数','requests in prefill'),cmax:0,pure:false});
  $('winstat').innerHTML=allT?`${tr('窗口平均','Window avg')} decode <b>${f0(allD/allT)}</b> · prefill <b>${f0(allP/allT)}</b> t/s ｜ ${tr('纯 decode ','pure decode ')}${f0(pureT)}${tr(' 秒',' s')} decode <b>${f0(pureT?pureD/pureT:null)}</b> ｜ ${tr('混合 ','mixed ')}${f0(mixT)}${tr(' 秒',' s')} decode <b>${f0(mixT?mixD/mixT:null)}</b> t/s ｜ ${tr('每点 ','per point ')}${bw>=60000?f1(bw/60000)+tr(' 分',' min'):f0(bw/1000)+tr(' 秒',' s')}`:tr('窗口内暂无数据','No data in window');
}
function drawCtxHist(d){
  // slider: 0=1k 1=2k 2=4k 3=8k per bar
  const bwEl=$('chbw'),cvv=$('chbwval');
  const bwIdx=bwEl?+bwEl.value:0,stride=1<<bwIdx;
  if(cvv)cvv.textContent=['1k','2k','4k','8k'][bwIdx];
  // support both old (6-bucket) and new (64-bucket) backends
  const raw=(d.ctx_hist||[]);
  const is64=raw.length>=32;
  const delta=raw.slice(); // cumulative since server start
  let merged;
  if(is64){
    // pad to 64
    while(delta.length<64)delta.push(0);
    const nB=Math.floor(64/stride);
    merged=[];for(let i=0;i<nB;i++){let s=0;for(let j=0;j<stride;j++)s+=delta[i*stride+j]||0;merged.push(s)}
  } else {
    // old 6-bucket fallback: map to roughly evenly-spaced bars
    merged=delta.slice(0,6);
  }
  const total=merged.reduce((a,c)=>a+c,0),cv=$('ch');
  if(!cv)return;
  const dpr=devicePixelRatio||1,W=cv.clientWidth,Hh=160;
  cv.height=Hh;cv.width=W*dpr;cv.height=Hh*dpr;
  cv.style.height=Hh+'px';
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,W,Hh);
  if(!total){
    g.fillStyle=css('--ink2');g.font='12px system-ui,sans-serif';g.textAlign='center';
    g.fillText(tr('暂无完成的请求','No completed requests yet'),W/2,Hh/2);$('chstat').textContent='';return;
  }
  const L=38,R=10,T=8,B=24,pw=W-L-R,ph=Hh-T-B;
  const nB=merged.length,mx=Math.max(1,...merged);
  // y-axis grid + labels
  const ys=niceStep(Math.max(1,mx/4));
  g.font='10px system-ui,sans-serif';
  g.textAlign='right';
  for(let yv=0;yv<=mx*1.05;yv+=ys){
    const y=T+ph*(1-yv/mx);if(y<T-4)break;
    g.fillStyle=css('--ink2');g.fillText(f0(yv),L-3,y+3);
    g.strokeStyle=css('--grid');g.lineWidth=0.5;g.beginPath();g.moveTo(L,y);g.lineTo(W-R,y);g.stroke();
  }
  // bars
  const bw=pw/nB;
  g.fillStyle=css('--dec');
  merged.forEach((v,i)=>{
    if(!v)return;
    const x=L+i*bw,bh=v/mx*ph,y=T+ph-bh;
    g.fillRect(x+0.5,Math.round(y),Math.max(1,bw-1.5),Math.ceil(bh));
  });
  // x-axis labels
  g.fillStyle=css('--ink2');g.textAlign='center';
  const labelEvery=is64?Math.max(1,Math.round(nB/8)):1;
  for(let i=0;i<=nB;i+=labelEvery){
    const lbl=is64?(i*stride+'k'):(['<4k','4k','8k','16k','32k','64k'][i]||'');
    g.fillText(lbl,L+i*bw,Hh-7);
  }
  $('chstat').textContent=tr(`累计完成 ${f0(total)} 个请求`,`${f0(total)} requests completed`);
  cv._hist={merged,stride,L,pw,nB,bw,T,ph,mx,is64};
}
function niceStep(x){const p=Math.pow(10,Math.floor(Math.log10(x||1))),m=x/p;return(m<=1?1:m<=2?2:m<=5?5:10)*p}
for(const cv of [$('tpd'),$('tpp')]){cv.addEventListener('mousemove',e=>{const c=cv._bins,tip=$('tip');if(!c)return;const x=e.clientX-cv.getBoundingClientRect().left;
  const i=Math.floor((c.t0+(x-c.L)/c.pw*(win*1000)-c.origin)/c.bw);
  if(i<0||i>=c.nb){tip.style.display='none';return}const b=c.bins[i];
  tip.innerHTML=`${new Date(c.origin+i*c.bw).toTimeString().slice(0,8)}<br>${c.name} <b>${!isNaN(c.rate[i])?f0(c.rate[i]):'–'}</b> t/s<br>${c.cl} <b>${!isNaN(c.cnt[i])?Math.round(c.cnt[i]):'–'}</b>${c.pure&&b.dt&&b.pure?`<br>${tr('纯 decode ','pure decode ')}${pct(b.pure/b.dt)}`:''}`;
  tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px'});
  cv.addEventListener('mouseleave',()=>$('tip').style.display='none')}
// histogram slider
if($('chbw'))$('chbw').addEventListener('input',()=>{if(last)drawCtxHist(last)});
// histogram hover tooltip
const ch=$('ch');
if(ch){ch.addEventListener('mousemove',e=>{const c=ch._hist,tip=$('tip');if(!c)return;
  const r=ch.getBoundingClientRect(),x=e.clientX-r.left;
  const i=Math.floor((x-c.L)/c.bw);
  if(i<0||i>=c.nB){tip.style.display='none';return}
  const v=c.merged[i],total=c.merged.reduce((a,b)=>a+b,0);
  const lo=c.is64?i*c.stride+'k':'';const hi=c.is64?(i+1)*c.stride+'k':'';
  tip.innerHTML=(c.is64?`${lo} – ${hi}<br>`:'')+'<b>'+f0(v)+'</b>'+tr(' 次',' requests')+( total?` (${pct(v/total)})`:'');
  tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px'});
  ch.addEventListener('mouseleave',()=>$('tip').style.display='none')}
addEventListener('resize',draw);
// a background tab polls slowly or not at all; on return fetch at once (history comes from the server)
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')poll()});
poll();setInterval(poll,1000);
</script></body></html>
)PFDASH";
}
