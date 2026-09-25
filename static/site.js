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
// Play the lead experiment silently while visible; preserve a visitor's pause.
// Reduced-motion preferences keep it still until the visitor presses play.
const heroVideo=document.getElementById('hero-video');
const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
let heroVisible=false, allowHeroPlayback=true;
function resumeHero(){
  if(heroVisible && allowHeroPlayback && !reducedMotion.matches && !document.hidden){
    heroVideo.play().catch(()=>{});
  }
}
heroVideo.addEventListener('pause',()=>{
  if(heroVisible && !document.hidden && !reducedMotion.matches) allowHeroPlayback=false;
});
heroVideo.addEventListener('play',()=>{allowHeroPlayback=true;});
if('IntersectionObserver' in window){
  const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{
    const visible=entry.isIntersecting && entry.intersectionRatio>=.25;
    if(entry.target===heroVideo){heroVisible=visible;if(visible) resumeHero();}
    if(!visible) entry.target.pause();
  }),{threshold:[0,.25]});
  document.querySelectorAll('video').forEach(video=>observer.observe(video));
}
document.addEventListener('visibilitychange',()=>{
  if(document.hidden) document.querySelectorAll('video').forEach(video=>video.pause());
  else resumeHero();
});
reducedMotion.addEventListener('change',()=>{
  if(reducedMotion.matches) heroVideo.pause();else resumeHero();
});

// A dedicated control keeps decorative background motion optional.
const heroMotion=document.getElementById('hero-motion');
function syncHeroControl(){
  const action=heroVideo.paused?'Play':'Pause';
  heroMotion.textContent=action+' background';
  heroMotion.setAttribute('aria-label',action+' background video');
}
heroVideo.addEventListener('play',syncHeroControl);
heroVideo.addEventListener('pause',syncHeroControl);
heroMotion.addEventListener('click',()=>{
  if(heroVideo.paused){allowHeroPlayback=true;heroVideo.play().catch(()=>{});}
  else{allowHeroPlayback=false;heroVideo.pause();}
});
syncHeroControl();
