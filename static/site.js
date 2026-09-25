'use strict';
// Exact aggregates from the archived x1 result files. No synthetic trajectories.
const conditions={
  loaded:{nominal:'14.43',dense:'0.027',sapc:'0.030',note:'With the payload attached, dense LS and SAPC reproduce the reference motion almost equally well.'},
  removed:{nominal:'2.60',dense:'11.73',sapc:'0.099',note:'After removal, mass corrections assigned to other links stay behind in the dense twin.'}
};
document.querySelectorAll('[data-condition]').forEach(button=>button.addEventListener('click',()=>{
  const state=button.dataset.condition, data=conditions[state];
  document.querySelectorAll('[data-condition]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
  for(const model of ['nominal','dense','sapc']){
    const node=document.getElementById(`${model}-score`);node.replaceChildren(document.createTextNode(data[model]));
    const unit=document.createElement('small');unit.textContent='mm';node.append(unit);
  }
  document.getElementById('condition-note').textContent=data.note;
}));
document.getElementById('copy-citation').addEventListener('click',async()=>{
  const text=document.getElementById('bibtex').textContent,status=document.getElementById('copy-status');
  try{await navigator.clipboard.writeText(text);status.textContent='BibTeX copied.';}
  catch{const range=document.createRange();range.selectNodeContents(document.getElementById('bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);status.textContent='Citation selected. Press Ctrl+C or ⌘C to copy.';}
});
// Pause media that are no longer visible; all research videos remain user-controlled.
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(!entry.isIntersecting)entry.target.pause();}),{threshold:.05});document.querySelectorAll('video').forEach(video=>observer.observe(video));}
