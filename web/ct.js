(() => {
  const $=id=>document.getElementById(id);
  let file=null, ready=false, result=null, previewURL=null, generation=0, controller=null;
  const update=()=>{$('run').disabled=!file||!ready||!$('confirm').checked||!!controller};
  const select=value=>{
    if(controller){controller.abort();controller=null;} generation++;
    result=null;$('report').hidden=true;$('scores').replaceChildren();$('prediction').textContent='Ready when you are';
    if(previewURL)URL.revokeObjectURL(previewURL);
    previewURL=null;file=null;$('preview').hidden=true;$('upload').value='';$('confirm').checked=false;
    if(!['image/jpeg','image/png','image/webp'].includes(value.type)||value.size>4*1024*1024||value.size===0){$('message').textContent='Choose a PNG, JPEG or WEBP under 4 MB.';update();return;}
    file=value;previewURL=URL.createObjectURL(file);$('preview').src=previewURL;$('preview').hidden=false;$('filename').textContent=file.name;
    $('message').textContent='Confirm that this is a de-identified CT image, then run the model.';$('run').textContent='Run trained model';update();
  };
  $('upload').addEventListener('change',e=>{if(e.target.files[0])select(e.target.files[0]);});
  $('confirm').onchange=update;
  $('drop').ondragover=e=>e.preventDefault();$('drop').ondrop=e=>{e.preventDefault();if(e.dataTransfer.files[0])select(e.dataTransfer.files[0]);};
  fetch('/api/health',{signal:AbortSignal.timeout(30000)}).then(async r=>{if(!r.ok)throw Error();return r.json();}).then(data=>{ready=data.ready;$('status').textContent=ready?'Trained CT model connected':'Model unavailable';update();}).catch(()=>{$('status').textContent='Model unavailable — refresh to retry';});
  $('run').onclick=async()=>{
    if(!file||!ready||!$('confirm').checked||controller)return;
    const token=generation;const selected=file;controller=new AbortController();const current=controller;const timer=setTimeout(()=>current.abort(),45000);
    $('run').textContent='Analyzing image…';$('prediction').textContent='Running saved model…';$('message').textContent='Processing your image.';$('scores').replaceChildren();$('report').hidden=true;result=null;update();
    try{
      const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':selected.type},body:selected,signal:current.signal});
      if(!response.headers.get('content-type')?.includes('application/json'))throw Error('The model service returned an unexpected response. Please retry.');
      const data=await response.json();if(!response.ok)throw Error(data.error||'Inference failed.');if(generation!==token)return;
      result={...data,file:selected.name,created_at:new Date().toISOString()};$('prediction').textContent='Predicted class: '+data.class;$('message').textContent=data.warning;
      Object.entries(data.scores).forEach(([label,score])=>{const row=document.createElement('div');row.className='score';const name=document.createElement('span');name.textContent=label;const value=document.createElement('strong');value.textContent=score.toFixed(3);row.append(name,value);$('scores').append(row);});
      const note=document.createElement('p');note.className='muted';note.textContent='Raw model scores, not percentages or cancer probabilities. Highest score selects the class.';$('scores').append(note);$('report').hidden=false;
    }catch(error){if(generation===token){$('prediction').textContent='Analysis unavailable';$('message').textContent=error.name==='AbortError'?'The request timed out. Please retry.':error.message;}}
    finally{clearTimeout(timer);if(controller===current){controller=null;$('run').textContent='Run trained model';update();}}
  };
  $('report').onclick=()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='lunglens-research-result.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  fetch('real_samples/samples.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(samples=>samples.forEach(sample=>{
    const card=document.createElement('article');card.className='sample';const img=document.createElement('img');img.src='real_samples/'+sample.file;img.alt='Original CT slice, dataset label '+sample.label;img.loading='lazy';
    const title=document.createElement('h3');title.textContent=sample.label+' example';const source=document.createElement('p');source.textContent=sample.original_file+' · held-out test image';
    const actions=document.createElement('div');actions.className='sample-actions';const use=document.createElement('button');use.textContent='Use this image';use.onclick=async()=>{use.disabled=true;try{const response=await fetch(img.src);if(!response.ok)throw Error();const blob=await response.blob();select(new File([blob],sample.file,{type:'image/jpeg'}));$('workspace').scrollIntoView({behavior:'smooth'});}catch{$('message').textContent='Could not load sample. Please retry.';}finally{use.disabled=false;}};
    const download=document.createElement('a');download.href=img.src;download.download=sample.file;download.textContent='Download ↓';actions.append(use,download);card.append(img,title,source,actions);$('gallery').append(card);
  })).catch(()=>{$('gallery').textContent='Sample images could not load. Please refresh.';});
  fetch('real_samples/evaluation.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(e=>{$('metrics').textContent=`Held-out image accuracy: ${(e.test.accuracy*100).toFixed(1)}% · Macro F1: ${e.test.macro_f1.toFixed(3)}. Train / validation / test: ${e.split_sizes.train} / ${e.split_sizes.validation} / ${e.split_sizes.test}. Not patient-independent validation.`;}).catch(()=>{$('metrics').textContent='Evaluation unavailable.';});
})();
