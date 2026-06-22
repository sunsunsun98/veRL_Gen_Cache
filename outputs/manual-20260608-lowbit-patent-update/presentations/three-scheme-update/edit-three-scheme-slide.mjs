import fs from "node:fs/promises";
import {
  createSlideContext,
  ensureArtifactToolWorkspace,
  importArtifactTool,
  saveBlobToFile,
} from "file:///C:/Users/12708/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs";

const source = process.argv[2];
const output = process.argv[3];
const preview = process.argv[4];
const workspace =
  "D:/Program Files (x86)/code_repo/veRL/outputs/manual-20260608-lowbit-patent-update/presentations/three-scheme-update";

if (!source || !output || !preview) {
  throw new Error("Usage: node edit-three-scheme-slide.mjs <source.pptx> <output.pptx> <preview.png>");
}

function items(collection) {
  if (!collection) return [];
  if (Array.isArray(collection.items)) return collection.items;
  if (Number.isInteger(collection.count) && typeof collection.getItem === "function") {
    return Array.from({ length: collection.count }, (_, index) => collection.getItem(index));
  }
  return [];
}

function shapeById(slide, id) {
  return items(slide.shapes).find((shape) => String(shape.id) === String(id));
}

function deleteShapeById(slide, id) {
  const shape = shapeById(slide, id);
  if (shape && typeof shape.delete === "function") {
    shape.delete();
    return;
  }
  try {
    slide.shapes.deleteById(id);
  } catch {}
}

function addText(ctx, slide, opts) {
  const {
    x,
    y,
    w,
    h,
    text,
    fill = "#00000000",
    line = "#00000000",
    size = 10,
    color = "#111111",
    bold = false,
    align = "center",
    valign = "middle",
    inset = 4,
  } = opts;
  const shape = ctx.addText(slide, {
    x,
    y,
    w,
    h,
    text,
    fontSize: size,
    color,
    bold,
    typeface: "Microsoft YaHei",
    align,
    valign,
    fill,
    line: { style: "solid", fill: line, width: line === "#00000000" ? 0 : 1.1 },
    insets: { left: inset, right: inset, top: Math.max(2, inset - 1), bottom: Math.max(2, inset - 1) },
  });
  shape.text.fontSize = size;
  shape.text.color = color;
  shape.text.bold = Boolean(bold);
  shape.text.typeface = "Microsoft YaHei";
  shape.text.alignment = align;
  shape.text.verticalAlignment = valign;
  shape.text.insets = { left: inset, right: inset, top: Math.max(2, inset - 1), bottom: Math.max(2, inset - 1) };
  return shape;
}

function box(ctx, slide, x, y, w, h, text, fill, line, size = 10, bold = false, color = "#111111") {
  return addText(ctx, slide, { x, y, w, h, text, fill, line, size, bold, color, inset: 6 });
}

function label(ctx, slide, x, y, w, h, text, size = 9, color = "#333333", bold = false, align = "center") {
  return addText(ctx, slide, { x, y, w, h, text, size, color, bold, align, inset: 1 });
}

function arrow(ctx, slide, x, y, color = "#444444") {
  return label(ctx, slide, x, y, 22, 18, "→", 14, color, true);
}

await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const slide = items(presentation.slides)[1];
const ctx = createSlideContext(null, {
  slideSize: { width: 1280, height: 720 },
  workspaceDir: workspace,
  titleFont: "Microsoft YaHei",
  bodyFont: "Microsoft YaHei",
});

for (const id of [35, 41, 49, 61, 64, 65, 69, 101, 102, 24581]) {
  deleteShapeById(slide, id);
  try {
    slide.images?.deleteById?.(id);
  } catch {}
}
for (const image of items(slide.images)) {
  if (["61", "69"].includes(String(image.id))) {
    try {
      image.delete();
    } catch {}
  }
}

label(
  ctx,
  slide,
  58,
  62,
  1160,
  42,
  "本发明涉及低比特强化学习后训练与推理加速的量化优化方法。在RL后训练中，模型需要反复执行rollout生成、log_p计算和策略更新，推理侧W8A8低比特路径在权重、激活与舍入环节引入的数值扰动，会导致训练侧伪量化模拟路径与推理侧真实量化路径分布不一致。针对上述问题，本文将量化尺度建模、舍入误差抑制和敏感层自适应回退统一纳入INT8 RL训推协同优化框架，主体方案如下：",
  10.2,
  "#111111",
  false,
  "left",
);
label(
  ctx,
  slide,
  36,
  112,
  1195,
  66,
  "1. 双尺度QAT协同感知：在QAT线性层伪量化路径中联合引入权重侧可学习weight scale与激活侧smooth scale。前者使权重量化步长随训练过程和权重动态范围自适应调整，后者按输入通道对激活与权重进行成对重参数化，将激活侧异常离群值平滑迁移至权重侧，在保持浮点计算等价的前提下降低A8量化动态范围压力。反向传播时，仅在有效量化区间传递梯度，对越界离群值施加边界约束，并按照量化上界与通道规模归一化更新幅度，避免离群值导致scale剧烈跳变和梯度振荡。",
  9.5,
  "#0070c0",
  true,
  "left",
);
label(
  ctx,
  slide,
  36,
  182,
  1195,
  48,
  "2. 哈希随机舍入：针对传统最近邻舍入会产生固定方向偏差、长序列rollout中误差容易累积的问题，将待量化数值的比特内容映射为确定性伪随机数，并与量化前小数残差比较确定向上或向下舍入方向。该机制使单次舍入误差在统计意义上趋于无偏，同时由于伪随机数由输入数值本身确定，训练侧伪量化与推理侧真实量化在相同输入下保持一致舍入结果，减少随机性造成的训推不一致。",
  9.5,
  "#0070c0",
  true,
  "left",
);
label(
  ctx,
  slide,
  36,
  233,
  1195,
  50,
  "3. 敏感层自适应回退：考虑不同层对INT8量化误差的敏感度存在显著差异，训练过程中实时统计各层权重和激活在伪量化前后的累计误差，并通过动量更新方式形成稳定的层级敏感度分数。对于连续多轮处于高误差区间的关键层，自动切换至更高精度路径执行rollout计算，从结构层面抑制敏感层量化噪声对策略分布的放大效应，在尽量保留INT8加速收益的同时提升低比特RL训练稳定性。",
  9.5,
  "#0070c0",
  true,
  "left",
);

box(ctx, slide, 30, 300, 1215, 226, "", "#ffffff", "#111111", 10);
box(ctx, slide, 30, 300, 1215, 28, "三类主题方案：双尺度QAT建模、哈希舍入抑偏、敏感层自适应回退", "#f2f2f2", "#111111", 14, true);

const colY = 338;
const colH = 174;
const colW = 383;
const gap = 18;
const x1 = 48;
const x2 = x1 + colW + gap;
const x3 = x2 + colW + gap;

box(ctx, slide, x1, colY, colW, colH, "", "#eef5ff", "#2f5597");
box(ctx, slide, x1 + 12, colY + 10, colW - 24, 26, "1. 双尺度QAT协同感知", "#d9eafc", "#2f5597", 12, true, "#0b4f9c");
box(ctx, slide, x1 + 20, colY + 50, 92, 38, "weight scale\n跟踪权重网格", "#ffffff", "#2f5597", 9.5, true);
arrow(ctx, slide, x1 + 116, colY + 60, "#2f5597");
box(ctx, slide, x1 + 142, colY + 50, 92, 38, "smooth scale\n平滑激活离群值", "#fff2cc", "#bf9000", 9.5, true);
arrow(ctx, slide, x1 + 238, colY + 60, "#2f5597");
box(ctx, slide, x1 + 264, colY + 50, 90, 38, "x/s, W*s\n伪量化", "#ffffff", "#2f5597", 9.5, true);
box(ctx, slide, x1 + 20, colY + 106, 154, 42, "有效区间传梯度\n越界值受边界约束", "#ffffff", "#2f5597", 9);
arrow(ctx, slide, x1 + 181, colY + 117, "#2f5597");
box(ctx, slide, x1 + 208, colY + 106, 146, 42, "按量化上界/通道规模\n归一化更新幅度", "#d9eafc", "#2f5597", 9, true);
label(ctx, slide, x1 + 55, colY + 154, 270, 14, "目标：抑制离群值导致的量化网格跳变与梯度振荡", 8.5, "#0b4f9c", true);

box(ctx, slide, x2, colY, colW, colH, "", "#fff8e8", "#bf9000");
box(ctx, slide, x2 + 12, colY + 10, colW - 24, 26, "2. 哈希随机舍入误差抑制", "#fff2cc", "#bf9000", 12, true, "#8a6200");
box(ctx, slide, x2 + 20, colY + 50, 82, 38, "浮点值\n小数残差", "#ffffff", "#bf9000", 9.5, true);
arrow(ctx, slide, x2 + 106, colY + 60, "#8a6200");
box(ctx, slide, x2 + 132, colY + 50, 98, 38, "比特哈希\n生成u", "#ffffff", "#bf9000", 9.5, true);
arrow(ctx, slide, x2 + 234, colY + 60, "#8a6200");
box(ctx, slide, x2 + 260, colY + 50, 94, 38, "比较u与残差\n决定舍入", "#fff2cc", "#bf9000", 9.2, true);
box(ctx, slide, x2 + 20, colY + 106, 150, 42, "相同输入\n训练/推理舍入一致", "#ffffff", "#bf9000", 9.2);
arrow(ctx, slide, x2 + 177, colY + 117, "#8a6200");
box(ctx, slide, x2 + 204, colY + 106, 150, 42, "误差期望趋零\n削弱系统性偏置", "#fff2cc", "#bf9000", 9.2, true);
label(ctx, slide, x2 + 58, colY + 154, 250, 14, "目标：长序列rollout中不累积固定方向舍入误差", 8.5, "#8a6200", true);

box(ctx, slide, x3, colY, colW, colH, "", "#fff1f0", "#c00000");
box(ctx, slide, x3 + 12, colY + 10, colW - 24, 26, "3. 敏感层精度自适应回退", "#fce4d6", "#c00000", 12, true, "#9c0000");
box(ctx, slide, x3 + 18, colY + 50, 96, 38, "实时统计\n层级量化误差", "#ffffff", "#c00000", 9.2, true);
arrow(ctx, slide, x3 + 118, colY + 60, "#9c0000");
box(ctx, slide, x3 + 144, colY + 50, 96, 38, "动量更新\n敏感度分数", "#ffffff", "#c00000", 9.2, true);
arrow(ctx, slide, x3 + 244, colY + 60, "#9c0000");
box(ctx, slide, x3 + 270, colY + 50, 84, 38, "阈值判定\n高敏感层", "#fce4d6", "#c00000", 9.2, true);
box(ctx, slide, x3 + 18, colY + 106, 150, 42, "Attention等关键层\n保持高精度路径", "#ffffff", "#c00000", 9.2);
arrow(ctx, slide, x3 + 176, colY + 117, "#9c0000");
box(ctx, slide, x3 + 204, colY + 106, 150, 42, "减少策略分布偏移\n稳定长序列训练", "#fce4d6", "#c00000", 9.2, true);
label(ctx, slide, x3 + 58, colY + 154, 250, 14, "目标：只回退真正敏感层，保留INT8加速收益", 8.5, "#9c0000", true);

box(ctx, slide, 30, 532, 1215, 114, "", "#ffc000", "#ffc000", 9);
label(ctx, slide, 80, 536, 1060, 16, "提出一种基于双尺度协同感知量化的INT8 RL训推优化方法。主要保护点如下：", 11.5, "#c00000", true, "left");

const pY = 555;
const pW = 370;
box(ctx, slide, 56, pY, pW, 24, "1：双尺度QAT量化尺度建模与梯度稳定更新方法", "#ffd966", "#c00000", 9.7, true, "#c00000");
label(ctx, slide, 60, pY + 29, pW - 8, 53, "联合构建weight scale与smooth scale；通过有效区间梯度、越界边界约束和归一化更新，使离群值存在时尺度仍稳定。", 9.4, "#111111", false, "left");

box(ctx, slide, 455, pY, pW, 24, "2：哈希随机舍入量化误差抑制方法", "#ffd966", "#c00000", 9.7, true, "#c00000");
label(ctx, slide, 459, pY + 29, pW - 8, 53, "由数值比特生成确定性伪随机数，与残差比较确定舍入方向，实现统计无偏和训推一致舍入。", 9.4, "#111111", false, "left");

box(ctx, slide, 854, pY, pW, 24, "3：累计量化误差动量估计的敏感层精度回退方法", "#ffd966", "#c00000", 9.2, true, "#c00000");
label(ctx, slide, 858, pY + 29, pW - 8, 53, "实时累计层级量化误差并以动量更新敏感度，对持续高敏感层回退精度，抑制rollout误差累积。", 9.4, "#111111", false, "left");

await fs.mkdir(preview.substring(0, preview.lastIndexOf("/")), { recursive: true }).catch(() => {});
const png = await presentation.export({ slide, format: "png", scale: 1 });
await saveBlobToFile(png, preview);

await fs.mkdir(output.substring(0, output.lastIndexOf("/")), { recursive: true }).catch(() => {});
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(output);
console.log(output);
