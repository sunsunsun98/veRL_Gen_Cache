import fs from "node:fs/promises";
import { ensureArtifactToolWorkspace, importArtifactTool, saveBlobToFile } from "file:///C:/Users/12708/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs";

const workspace = "D:/Program Files (x86)/code_repo/veRL/outputs/manual-20260605-lowbit-patent-rev/presentations/dual-scale-int8-rl-rev";
const source = `${workspace}/source-current.pptx`;
const output = `${workspace}/output/revised.pptx`;
const previewDir = `${workspace}/preview`;
const layoutDir = `${workspace}/layout`;

function collectionItems(collection) {
  if (!collection) return [];
  if (Array.isArray(collection.items)) return collection.items;
  if (Number.isInteger(collection.count) && typeof collection.getItem === "function") {
    return Array.from({ length: collection.count }, (_, index) => collection.getItem(index));
  }
  return [];
}

function shapeById(slide, id) {
  return collectionItems(slide.shapes).find((shape) => String(shape.id) === String(id));
}

function imageById(slide, id) {
  return collectionItems(slide.images).find((image) => String(image.id) === String(id));
}

function setText(slide, id, text) {
  const shape = shapeById(slide, id);
  if (!shape) throw new Error(`Missing shape ${id}`);
  shape.text = text;
}

await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const slides = collectionItems(presentation.slides);
const slide = slides[1];

setText(slide, 24581, [
  "本发明涉及低比特强化学习后训练与推理加速的量化优化方法。针对W8A8 rollout中权重量化网格失配、激活离群值放大及舍入偏差累积导致的训推分布不一致问题，提出一种基于双尺度协同感知量化的INT8 RL训推优化方法，主体方案如下：",
  "权重量化尺度自适应感知模块：在线性层权重量化路径中引入可学习weight scale，将量化步长纳入QAT优化；前向模拟INT8映射与反量化误差，反向经STE感知截断和重构误差，并结合动态范围跟踪与LSQ梯度缩放，实现尺度稳定更新。",
  "激活-权重双向平滑重参数化模块：针对激活侧通道离群值，引入按输入通道定义的smooth scale，对线性层输入执行x/s缩放、对权重执行W·s补偿，在保持浮点计算等价的前提下降低W8A8激活量化动态范围压力。",
  "训推一致性量化偏差协同校准模块：联合敏感层精度回退、哈希随机舍入和scale稳定化更新，对舍入偏差、敏感层误差和量化残差进行协同校准，抑制长序列rollout中的量化误差累积。",
].join("\n"));

setText(slide, 35, [
  "提出一种基于双尺度协同感知量化的INT8 RL训推优化方法。主要保护点如下：",
  "1：保护一种双尺度协同感知的QAT量化尺度建模方法",
  "其特征在于联合构建可学习weight scale与smooth scale两类尺度变量，分别对权重量化步长和激活离群值进行协同建模，并通过动态范围跟踪与梯度缩放提升scale更新稳定性。",
  "2：保护一种面向INT8 RL训推一致性的量化偏差校准方法",
  "其特征在于结合敏感层精度回退、哈希随机舍入和scale稳定化更新，对W8A8 rollout中的舍入偏差、敏感层误差和量化残差进行协同校准，提升低比特RL训练稳定性。",
].join("\n"));

setText(slide, 42, "权重量化尺度自适应感知模块");
setText(slide, 74, "激活-权重双向平滑重参数化模块");
setText(slide, 87, "weight scale随训练阶段稳定调整");
setText(slide, 100, "smooth scale实现离群值平滑迁移");

// Re-apply the main body with compact wording so the inherited numbered
// paragraphs do not collide with the figure area.
setText(slide, 24581, [
  "本发明涉及低比特强化学习后训练与推理加速的量化优化方法。针对W8A8 rollout中权重量化网格失配、激活离群值放大及舍入偏差累积导致的训推分布不一致问题，提出一种基于双尺度协同感知量化的INT8 RL训推优化方法，主体方案如下：",
  "权重量化尺度自适应感知模块：在线性层权重量化路径中引入可学习weight scale，将量化步长纳入QAT优化；前向模拟INT8映射与反量化误差，反向经STE感知截断和重构误差，并结合动态范围跟踪与LSQ梯度缩放，实现尺度稳定更新。",
  "激活-权重双向平滑重参数化模块：针对激活侧通道离群值，引入按输入通道定义的smooth scale，对线性层输入执行x/s缩放、对权重执行W·s补偿，在保持浮点计算等价的前提下降低W8A8激活量化动态范围压力。",
  "量化偏差协同校准模块：联合敏感层精度回退、哈希随机舍入和scale稳定化更新，对舍入偏差、敏感层误差和量化残差进行校准，抑制长序列rollout中的量化误差累积。",
].join("\n"));

const flowSvg = `${workspace}/flow-bias-calib.svg`;
await fs.writeFile(
  flowSvg,
  `<svg xmlns="http://www.w3.org/2000/svg" width="520" height="150" viewBox="0 0 520 150">
    <rect width="520" height="150" fill="white"/>
    <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#333"/></marker></defs>
    <rect x="20" y="30" width="120" height="44" rx="4" fill="#e9f3ff" stroke="#2f5597" stroke-width="2"/>
    <text x="80" y="57" font-size="16" text-anchor="middle" font-family="Arial" fill="#111">Weight Scale</text>
    <rect x="200" y="30" width="120" height="44" rx="4" fill="#fff2cc" stroke="#bf9000" stroke-width="2"/>
    <text x="260" y="57" font-size="16" text-anchor="middle" font-family="Arial" fill="#111">Smooth Scale</text>
    <rect x="380" y="30" width="120" height="44" rx="4" fill="#fce4d6" stroke="#c00000" stroke-width="2"/>
    <text x="440" y="57" font-size="16" text-anchor="middle" font-family="Arial" fill="#111">Bias Calib.</text>
    <line x1="142" y1="52" x2="195" y2="52" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <line x1="322" y1="52" x2="375" y2="52" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <text x="80" y="112" font-size="14" text-anchor="middle" font-family="Arial" fill="#333">量化尺度自适应</text>
    <text x="260" y="112" font-size="14" text-anchor="middle" font-family="Arial" fill="#333">离群值平滑迁移</text>
    <text x="440" y="112" font-size="14" text-anchor="middle" font-family="Arial" fill="#333">量化偏差校准</text>
  </svg>`,
  "utf8",
);
const image = imageById(slide, 24582);
if (image && typeof image.replace === "function") {
  try {
    await image.replace(await FileBlob.load(flowSvg));
  } catch {}
}

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(layoutDir, { recursive: true });
for (let index = 0; index < slides.length; index += 1) {
  const padded = String(index + 1).padStart(2, "0");
  const png = await presentation.export({ slide: slides[index], format: "png", scale: 1 });
  await saveBlobToFile(png, `${previewDir}/slide-${padded}.png`);
  const layout = await presentation.export({ slide: slides[index], format: "layout" });
  await saveBlobToFile(layout, `${layoutDir}/slide-${padded}.layout.json`);
}

await fs.mkdir(`${workspace}/output`, { recursive: true });
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(output);
console.log(output);
