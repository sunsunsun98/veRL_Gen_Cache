import { ensureArtifactToolWorkspace, importArtifactTool } from 'file:///C:/Users/12708/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs';
const workspace = 'D:/Program Files (x86)/code_repo/veRL/outputs/manual-20260605-lowbit-patent-rev/presentations/dual-scale-int8-rl-rev';
await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(await FileBlob.load(`${workspace}/source-current.pptx`));
const slides = Array.isArray(presentation.slides?.items) ? presentation.slides.items : Array.from({length:presentation.slides.count},(_,i)=>presentation.slides.getItem(i));
function items(c){return Array.isArray(c?.items)?c.items:Array.from({length:c?.count||0},(_,i)=>c.getItem(i));}
for (const el of items(slides[1].shapes)) { const t=String(el.text??'').replace(/\n/g,' | '); if(t.trim()) console.log(el.id, el.name, t.slice(0,160)); }
