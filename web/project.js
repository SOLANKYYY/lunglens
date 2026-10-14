/* Faculty project extension: educational stages, synthetic downloads, Python inference.
   Same-origin /api routes work when served by python -m backend.server.
   Vercel hosts the static frontend; it does not magically start the Python server. */
(() => {
  const get = id => document.getElementById(id);
  const stageData = [
    ['0','In situ','Abnormal cells are confined to the airway lining. This is called carcinoma in situ.'],
    ['I','Localized','Cancer is present in the lung and has not spread to lymph nodes. Exact substages depend on tumor characteristics.'],
    ['II','More local extent','A larger or more locally involved tumor, and/or spread to certain nearby lymph nodes, can meet Stage II criteria.'],
    ['III','Locally advanced','More extensive involvement within the chest, including certain regional lymph nodes or nearby structures. Specific TNM combinations define this stage.'],
    ['IV','Metastatic','Cancer has spread in ways that meet metastatic criteria, such as to distant organs. Specific patterns of spread determine substages.']
  ];
  document.querySelectorAll('[data-stage]').forEach(button => button.onclick = () => {
    const [number,title,description] = stageData[Number(button.dataset.stage)];
    get('stage-number').textContent=number;get('stage-title').textContent=`Stage ${number} · ${title}`;get('stage-description').textContent=description;
    document.querySelectorAll('[data-stage]').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-pressed',String(b===button))});
  });
  document.querySelectorAll('[data-demo-file]').forEach(button=>button.onclick=()=>{
    const a=document.createElement('a');a.href=`demo_samples/${button.dataset.demoFile}.png`;a.download=`synthetic-${button.dataset.demoFile}.png`;a.click();
  });
  let model=null,selectedFile=null,request=null,result=null;
  const reset=()=>{if(request)request.abort();request=null;selectedFile=null;result=null;get('run-local').hidden=true;get('run-local').disabled=false;get('run-local').textContent='Run local model';get('local-result').hidden=true;get('export-local').hidden=true};
  window.addEventListener('scan-reset',reset);
  window.addEventListener('local-image-ready',e=>{selectedFile=e.detail.file;get('run-local').hidden=!model});
  fetch('/api/health',{signal:AbortSignal.timeout(3500)}).then(r=>{if(!r.ok)throw Error();return r.json()}).then(data=>{
    if(!data.ready)throw Error('No model');model=data.metadata;
    get('connection-label').textContent='LOCAL PYTHON API CONNECTED';
    get('connection-title').textContent=`${model.framework} model · ${model.data_kind==='synthetic'?'Artificial patterns only':'Unvalidated research'}`;
    get('connection-info').textContent='Uploads are previewed locally. The “Run local model” button sends the image to this Python server for inference. No stage is predicted.';
    if(selectedFile)get('run-local').hidden=false;
    const status=document.querySelector('.status-text');status.replaceChildren(document.createTextNode('Local model'));const em=document.createElement('em');em.textContent=model.data_kind==='synthetic'?'synthetic patterns only':'research only';status.append(em);
  }).catch(()=>{get('connection-label').textContent='BROWSER DEMO';get('connection-title').textContent='Start Python for model inference';get('connection-info').textContent='This page still supports sample simulations and image previews. From VS Code, run: python -m backend.server, then open http://127.0.0.1:8000.'});
  get('run-local').onclick=async()=>{
    if(!selectedFile||!model||request)return;
    const controller=new AbortController();request=controller;const timer=setTimeout(()=>controller.abort(),30000);
    get('run-local').disabled=true;get('run-local').textContent='Running local model…';get('local-result').hidden=true;get('export-local').hidden=true;
    try{
      const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':selectedFile.type},body:selectedFile,signal:controller.signal});
      const output=await response.json();if(!response.ok)throw Error(output.error||'Inference failed');if(request!==controller)return;
      result={...output,file:selectedFile.name,generated:new Date().toISOString()};const box=get('local-result');box.replaceChildren();box.hidden=false;
      const title=document.createElement('h4');title.textContent=`${output.data_kind==='synthetic'?'Artificial pattern':'Research class'}: ${output.class}`;box.append(title);
      for(const [label,score] of Object.entries(output.scores)){const row=document.createElement('div');row.className='score-row';const name=document.createElement('span');name.textContent=label;const bar=document.createElement('progress');bar.max=1;bar.value=score;bar.setAttribute('aria-label',`${label} model score`);const value=document.createElement('b');value.textContent=(score*100).toFixed(1)+'%';row.append(name,bar,value);box.append(row)}
      for(const text of ['Class scores, not cancer risk or stage.',output.warning,'Cancer stage: Not assessed.']){const p=document.createElement('p');p.textContent=text;box.append(p)}
      get('export-local').hidden=false;
    }catch(error){if(request!==controller)return;const box=get('local-result');box.hidden=false;box.textContent=error.name==='AbortError'?'Model request timed out. Check the Python terminal and retry.':error.message}
    finally{clearTimeout(timer);if(request===controller){request=null;get('run-local').disabled=false;get('run-local').textContent='Run local model'}}
  };
  get('export-local').onclick=()=>{
    if(!result)return;const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='lunglens-research-result.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
})();
