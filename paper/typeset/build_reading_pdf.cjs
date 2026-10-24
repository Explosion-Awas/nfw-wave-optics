/* Build an offline mathematical reading edition, not a journal class file.
 * APIs: https://katex.org/docs/api
 * https://playwright.dev/docs/api/class-page#page-pdf
 * Usage: node build_reading_pdf.cjs --modules-root PATH --chrome PATH
 */
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const crypto = require('crypto');
const args = process.argv.slice(2);
function arg(name, fallback) { const i=args.indexOf(name); return i>=0?args[i+1]:fallback; }
const root = path.resolve(__dirname, '..');
const modulesRoot = arg('--modules-root', root);
const chrome = arg('--chrome');
if (!chrome || !fs.existsSync(chrome)) throw Error('Supply an installed Chromium executable with --chrome');
const resolve = (name) => require.resolve(name, {paths:[modulesRoot]});
const katex = require(resolve('katex'));
const {chromium} = require(resolve('playwright'));
const hash = text => crypto.createHash('sha256').update(text).digest('hex');
const sourcePath = path.join(root,'manuscript_cqg_en.md');
const source = fs.readFileSync(sourcePath,'utf8');
const title = source.split('\n')[0].replace(/^# /,'');
const escape = s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
(async()=>{
  const {marked} = await import(pathToFileURL(resolve('marked')).href);
  const renderedMath=[];
  const warnings=[];
  function protect(tex, displayMode){
    const output=katex.renderToString(tex,{displayMode,throwOnError:true,strict:(code,message)=>{warnings.push({code,message});return 'warn';},trust:false,output:'htmlAndMathml'});
    const i=renderedMath.length;
    renderedMath.push({tex,displayMode,output});
    return `NFWPROTECTEDMATH${i}END`;
  }
  let input=source.replace(/\\\[([\s\S]*?)\\\]/g,(_,t)=>'\n\n'+protect(t,true)+'\n\n');
  input=input.replace(/\\\(([\s\S]*?)\\\)/g,(_,t)=>protect(t,false));
  input=input.replace(/\$([^\n$]+?)\$/g,(_,t)=>protect(t,false));
  let html=marked.parse(input,{gfm:true,breaks:false});
  html=html.replace(/<p>(NFWPROTECTEDMATH\d+END)<\/p>/g,'$1');
  html=html.replace(/NFWPROTECTEDMATH(\d+)END/g,(_,n)=>{
    const m=renderedMath[Number(n)];
    return m.displayMode?`<div class="math-block">${m.output}</div>`:m.output;
  });
  if (/NFWPROTECTEDMATH\d+END/.test(html)) throw Error('Unexpanded mathematics');
  html=html.replace('<h2>Sources</h2>','<h2>References</h2>');
  // Keep reference identities, titles and URLs unchanged; provide paragraph spacing.
  const refIndex=html.indexOf('<h2>References</h2>');
  if(refIndex<0) throw Error('Missing references');
  const before=html.slice(0,refIndex);
  let refs=html.slice(refIndex).replace(/\n(?=\[\d+\])/g,'<br>\n');
  refs=refs.replace(/\[(\d+)\]/g,(_,n)=>`<span class="ref-number" id="ref-${n}">[${n}]</span>`);
  html=before.replace(/\[(\d+)\]/g,(_,n)=>`<a class="citation" href="#ref-${n}">[${n}]</a>`)+refs;
  const tableCaptions=['Analytical comparison definitions','Common-band parameters','Common-set accuracy at the prescribed targets','Integrated diffraction-residual error','Paired preparation-plus-first-band costs','Reuse under parameter changes'];
  let tableIndex=0;
  html=html.replace(/<table>/g,()=>`<table><caption>Table ${tableIndex+1}. ${tableCaptions[tableIndex++]}</caption>`);
  if(tableIndex!==tableCaptions.length) throw Error('Unexpected table count');
  const kd=path.dirname(resolve('katex'));
  const assetDir=path.join(root,'assets','katex');
  fs.mkdirSync(assetDir,{recursive:true});
  fs.copyFileSync(path.join(kd,'katex.min.css'),path.join(assetDir,'katex.min.css'));
  fs.cpSync(path.join(kd,'fonts'),path.join(assetDir,'fonts'),{recursive:true});
  fs.copyFileSync(path.join(kd,'..','LICENSE'),path.join(assetDir,'LICENSE'));
  const document=`<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${escape(title)}</title>
<link rel="stylesheet" href="assets/katex/katex.min.css">
<style>
@page { size:A4; margin:18mm 18mm 20mm; }
* { box-sizing:border-box; }
html { color:#111; background:#fff; }
body { margin:0; font:10.5pt/1.32 'Times New Roman',Times,serif; }
article { width:174mm; margin:0 auto; }
h1 { font-size:18pt; line-height:1.16; font-weight:600; text-align:center; margin:0 0 18pt; }
h2 { font-size:13pt; line-height:1.2; margin:18pt 0 7pt; break-after:avoid; }
h3 { font-size:11.2pt; line-height:1.2; margin:12pt 0 6pt; break-after:avoid; }
h4 { font-size:10.5pt; font-style:italic; margin:10pt 0 5pt; break-after:avoid; }
p { margin:0 0 8pt; text-align:justify; orphans:3; widows:3; }
p:has(+ .math-block), p:has(+ table) { break-after:avoid; }
.math-block { margin:8pt 0; padding:0; break-inside:avoid; page-break-inside:avoid; }
.katex { font-size:1.02em; }
.katex-display { margin:0; }
.katex-display>.katex { text-align:center; }
table { width:100%; border-collapse:collapse; margin:10pt 0 12pt; font-size:8.3pt; line-height:1.2; break-inside:avoid; }
th,td { padding:5pt 4pt; border-bottom:0.45pt solid #bbb; vertical-align:top; }
th { border-top:0.7pt solid #222; border-bottom:0.7pt solid #222; font-weight:600; }
thead { display:table-header-group; }
tr { break-inside:avoid; }
a { color:#111; text-decoration:none; }
a[href^="http"] { overflow-wrap:anywhere; }
.citation { white-space:nowrap; }
code { font:8pt/1.2 Consolas,monospace; overflow-wrap:anywhere; }
strong { font-weight:600; }
.ref-number { white-space:nowrap; }
@media screen { body { padding:28px 0; } article { padding:5mm 0; } }
@media print { article { margin:0; width:100%; } a { color:#111; } }
</style></head><body><article>${html}</article></body></html>`;
  const htmlPath=path.join(root,'manuscript_cqg_en.html');
  fs.writeFileSync(htmlPath,document,'utf8');
  fs.writeFileSync(path.join(root,'typeset/math_inventory.json'),JSON.stringify(renderedMath.map(({tex,displayMode})=>({tex,displayMode})),null,2));
  const browser=await chromium.launch({executablePath:chrome,headless:true,args:['--disable-gpu']});
  try {
    const page=await browser.newPage({viewport:{width:1100,height:1400},deviceScaleFactor:1});
    const networkErrors=[];
    page.on('requestfailed',request=>networkErrors.push({url:request.url(),failure:request.failure()}));
    await page.goto(pathToFileURL(htmlPath).href,{waitUntil:'load'});
    await page.evaluate(()=>document.fonts.ready);
    await page.emulateMedia({media:'print'});
    const layout=await page.evaluate(()=>{
      const shrunk=[];
      for(const [i,el] of [...document.querySelectorAll('.math-block')].entries()){
        let tries=0;
        while(el.scrollWidth>el.clientWidth+1&&tries<8){
          const old=parseFloat(getComputedStyle(el).fontSize);
          el.style.fontSize=(old*Math.min(.96,el.clientWidth/el.scrollWidth*.98))+'px';tries++;
        }
        if(tries) shrunk.push({index:i,fontSize:getComputedStyle(el).fontSize});
      }
      const overflow=[...document.querySelectorAll('.math-block,table')].map((el,i)=>({index:i,tag:el.tagName,width:el.clientWidth,scroll:el.scrollWidth})).filter(x=>x.scroll>x.width+2);
      const missingReferences=[...document.querySelectorAll('.citation')].filter(a=>!document.querySelector(a.getAttribute('href'))).map(a=>a.textContent);
      const strayTextNodes=[...document.querySelector('article').childNodes].filter(n=>n.nodeType===3&&n.textContent.trim()).map(n=>n.textContent.trim());
      return {shrunk,overflow,missingReferences,strayTextNodes,displayMath:document.querySelectorAll('.math-block').length,tables:document.querySelectorAll('table').length,mathErrors:document.querySelectorAll('.katex-error').length,bodyTextLength:document.body.innerText.length};
    });
    if(layout.overflow.length||layout.mathErrors||layout.missingReferences.length||layout.strayTextNodes.length||networkErrors.length) throw Error(JSON.stringify({layout,networkErrors}));
    // Persist the verified sizing for the offline HTML edition as well.
    const finalHTML=await page.content();
    fs.writeFileSync(htmlPath,finalHTML,'utf8');
    const pdfPath=path.join(root,'manuscript_cqg_en.pdf');
    await page.pdf({path:pdfPath,format:'A4',preferCSSPageSize:true,printBackground:true,displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="width:100%;font:9px Times New Roman;text-align:center;color:#555"><span class="pageNumber"></span></div>',margin:{top:'18mm',bottom:'20mm',left:'18mm',right:'18mm'},tagged:true});
    await page.emulateMedia({media:'screen'});
    await page.screenshot({path:path.join(root,'typeset/first_screen.png'),fullPage:false});
    const result={status:'PASS',renderer:'Chromium print from static KaTeX HTML (not a LaTeX compilation)',katex:katex.version,source_sha256:hash(source),pdf_sha256:hash(fs.readFileSync(pdfPath)),pdf_bytes:fs.statSync(pdfPath).size,html_bytes:fs.statSync(htmlPath).size,math_blocks:renderedMath.length,display_math:renderedMath.filter(x=>x.displayMode).length,layout,warnings,networkErrors};
    fs.writeFileSync(path.join(root,'typeset/render_receipt.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result,null,2));
  } finally {await browser.close();}
})().catch(err=>{console.error(err);process.exitCode=1;});
