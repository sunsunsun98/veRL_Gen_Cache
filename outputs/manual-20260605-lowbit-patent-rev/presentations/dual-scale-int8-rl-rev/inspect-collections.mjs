import { ensureArtifactToolWorkspace, importArtifactTool } from 'file:///C:/Users/12708/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs';
const workspace = 'D:/Program Files (x86)/code_repo/veRL/outputs/manual-20260605-lowbit-patent-rev/presentations/dual-scale-int8-rl-rev';
await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(await FileBlob.load(`${workspace}/source-current.pptx`));
const slide = (Array.isArray(presentation.slides?.items) ? presentation.slides.items : Array.from({length:presentation.slides.count},(_,i)=>presentation.slides.getItem(i)))[1];
for (const prop of ['shapes','elements']) {
 const c=slide[prop]; console.log(prop, c?.constructor?.name); let p=c,l=0; while((p=Object.getPrototypeOf(p))&&l<3){console.log(Object.getOwnPropertyNames(p));l++;}
}
