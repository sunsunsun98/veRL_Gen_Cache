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
  "D:/Program Files (x86)/code_repo/veRL/outputs/manual-20260605-lowbit-patent-rev/presentations/dual-scale-int8-rl-rev";

if (!source || !output || !preview) {
  throw new Error("Usage: node edit-qat-scale-flow.mjs <source.pptx> <output.pptx> <preview.png>");
}

function collectionItems(collection) {
  if (!collection) return [];
  if (Array.isArray(collection.items)) return collection.items;
  if (Number.isInteger(collection.count) && typeof collection.getItem === "function") {
    return Array.from({ length: collection.count }, (_, index) => collection.getItem(index));
  }
  return [];
}

function setTextStyle(shape, { size = 11, color = "#111111", bold = false, align = "center" } = {}) {
  shape.text.fontSize = size;
  shape.text.color = color;
  shape.text.bold = Boolean(bold);
  shape.text.typeface = "Microsoft YaHei";
  shape.text.alignment = align;
  shape.text.verticalAlignment = "middle";
  shape.text.insets = { left: 6, right: 6, top: 3, bottom: 3 };
}

function addBox(ctx, slide, { x, y, w, h, text, fill, line = "#2f5597", size = 11, bold = false, color = "#111111" }) {
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
    align: "center",
    valign: "middle",
    fill,
    line: { style: "solid", fill: line, width: 1.2 },
    insets: { left: 6, right: 6, top: 3, bottom: 3 },
  });
  setTextStyle(shape, { size, color, bold });
  return shape;
}

function addLabel(ctx, slide, { x, y, w, h, text, size = 10, color = "#333333", bold = false, align = "center" }) {
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
    valign: "middle",
    fill: "#00000000",
    line: { style: "solid", fill: "#00000000", width: 0 },
    insets: { left: 1, right: 1, top: 1, bottom: 1 },
  });
  setTextStyle(shape, { size, color, bold, align });
  return shape;
}

function addArrow(ctx, slide, x, y, text = "->", color = "#333333") {
  return addLabel(ctx, slide, { x, y, w: 30, h: 22, text, size: 16, color, bold: true });
}

await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const slides = collectionItems(presentation.slides);
const slide = slides[1];
const ctx = createSlideContext(null, {
  slideSize: { width: 1280, height: 720 },
  workspaceDir: workspace,
  titleFont: "Microsoft YaHei",
  bodyFont: "Microsoft YaHei",
});

for (const id of [41, 42, 49, 74, 87, 100, 101, 102]) {
  try {
    slide.shapes.deleteById(id);
  } catch {
    // Shape may not exist in a later user revision.
  }
}

const bodyShape = collectionItems(slide.shapes).find((shape) => String(shape.id) === "24581");
if (bodyShape) {
  bodyShape.text = [
    "本发明涉及低比特RL后训练与推理加速的量化优化方法。针对W8A8 rollout中权重量化尺度难以自适应、激活离群值放大量化动态范围以及训推分布不一致问题，提出基于双尺度协同感知量化的INT8 RL训推优化方法，主体方案如下：",
    "权重量化尺度自适应：引入可学习weight scale，将量化步长纳入QAT优化，并结合动态范围跟踪与LSQ梯度缩放，使weight scale随训练过程稳定更新。",
    "激活-权重双向平滑：引入按输入通道定义的smooth scale，对输入执行x/s、对权重执行W*s，在保持浮点等价的前提下降低激活动态范围压力。",
    "量化偏差协同校准：结合敏感层精度回退、哈希随机舍入和scale稳定化更新，校准W8A8 rollout中的舍入偏差与量化残差，抑制长序列误差累积。",
  ].join("\n");
}

addBox(ctx, slide, {
  x: 30,
  y: 302,
  w: 1215,
  h: 224,
  text: "",
  fill: "#ffffff",
  line: "#111111",
});
addBox(ctx, slide, {
  x: 30,
  y: 302,
  w: 1215,
  h: 28,
  text: "QAT伪量化过程中的 smooth scale 与 weight scale 更新逻辑",
  fill: "#f2f2f2",
  line: "#111111",
  size: 15,
  bold: true,
});

addBox(ctx, slide, {
  x: 48,
  y: 340,
  w: 575,
  h: 76,
  text: "",
  fill: "#eef5ff",
  line: "#2f5597",
});
addLabel(ctx, slide, {
  x: 58,
  y: 346,
  w: 180,
  h: 18,
  text: "weight scale更新闭环",
  size: 12,
  color: "#0b4f9c",
  bold: true,
  align: "left",
});

const blue = "#d9eafc";
addBox(ctx, slide, { x: 65, y: 371, w: 78, h: 30, text: "W_fp", fill: blue, line: "#2f5597", size: 11, bold: true });
addArrow(ctx, slide, 148, 375, "→", "#2f5597");
addBox(ctx, slide, {
  x: 176,
  y: 366,
  w: 118,
  h: 40,
  text: "伪量化\nround/clamp",
  fill: "#ffffff",
  line: "#2f5597",
  size: 10,
});
addArrow(ctx, slide, 299, 375, "→", "#2f5597");
addBox(ctx, slide, {
  x: 327,
  y: 366,
  w: 96,
  h: 40,
  text: "反量化\nW_hat",
  fill: "#ffffff",
  line: "#2f5597",
  size: 10,
});
addArrow(ctx, slide, 428, 375, "→", "#2f5597");
addBox(ctx, slide, {
  x: 456,
  y: 366,
  w: 145,
  h: 40,
  text: "STE反传 + LSQ\n梯度缩放更新s_w",
  fill: "#d9eafc",
  line: "#2f5597",
  size: 10,
  bold: true,
});

addBox(ctx, slide, {
  x: 48,
  y: 426,
  w: 575,
  h: 86,
  text: "",
  fill: "#fff6db",
  line: "#bf9000",
});
addLabel(ctx, slide, {
  x: 58,
  y: 432,
  w: 190,
  h: 18,
  text: "smooth scale更新闭环",
  size: 12,
  color: "#8a6200",
  bold: true,
  align: "left",
});

const yellow = "#fff2cc";
addBox(ctx, slide, { x: 65, y: 459, w: 86, h: 32, text: "激活x\n权重W", fill: yellow, line: "#bf9000", size: 10 });
addArrow(ctx, slide, 156, 464, "→", "#8a6200");
addBox(ctx, slide, {
  x: 184,
  y: 454,
  w: 102,
  h: 42,
  text: "s_s = exp(theta)\n按通道平滑",
  fill: "#ffffff",
  line: "#bf9000",
  size: 9.5,
});
addArrow(ctx, slide, 291, 464, "→", "#8a6200");
addBox(ctx, slide, {
  x: 319,
  y: 454,
  w: 108,
  h: 42,
  text: "x / s_s\nW * s_s",
  fill: "#ffffff",
  line: "#bf9000",
  size: 10,
  bold: true,
});
addArrow(ctx, slide, 432, 464, "→", "#8a6200");
addBox(ctx, slide, {
  x: 460,
  y: 454,
  w: 142,
  h: 42,
  text: "动态范围迁移\nQAT loss反传更新theta",
  fill: yellow,
  line: "#bf9000",
  size: 9.5,
});

addBox(ctx, slide, {
  x: 646,
  y: 340,
  w: 580,
  h: 172,
  text: "",
  fill: "#fbfbfb",
  line: "#777777",
});
addLabel(ctx, slide, {
  x: 660,
  y: 346,
  w: 270,
  h: 20,
  text: "双尺度协同感知：平滑后权重驱动量化尺度稳定更新",
  size: 12,
  color: "#333333",
  bold: true,
  align: "left",
});

addBox(ctx, slide, {
  x: 675,
  y: 379,
  w: 132,
  h: 42,
  text: "smooth scale\n产生W * s_s分布",
  fill: "#fff2cc",
  line: "#bf9000",
  size: 10,
});
addArrow(ctx, slide, 818, 389, "→", "#555555");
addBox(ctx, slide, {
  x: 848,
  y: 379,
  w: 140,
  h: 42,
  text: "EMA跟踪\n平滑后动态范围",
  fill: "#eeeeee",
  line: "#777777",
  size: 10,
});
addArrow(ctx, slide, 999, 389, "→", "#555555");
addBox(ctx, slide, {
  x: 1029,
  y: 379,
  w: 166,
  h: 42,
  text: "辅助weight scale\n避免网格跳变",
  fill: "#d9eafc",
  line: "#2f5597",
  size: 10,
  bold: true,
});

addBox(ctx, slide, {
  x: 675,
  y: 448,
  w: 160,
  h: 40,
  text: "前向：模拟W8A8 rollout噪声",
  fill: "#ffffff",
  line: "#999999",
  size: 10,
});
addArrow(ctx, slide, 846, 457, "→", "#555555");
addBox(ctx, slide, {
  x: 877,
  y: 448,
  w: 146,
  h: 40,
  text: "反向：截断掩码\n重构误差",
  fill: "#ffffff",
  line: "#999999",
  size: 10,
});
addArrow(ctx, slide, 1034, 457, "→", "#555555");
addBox(ctx, slide, {
  x: 1065,
  y: 448,
  w: 130,
  h: 40,
  text: "训推一致\n误差不累积",
  fill: "#fce4d6",
  line: "#c00000",
  size: 10,
  bold: true,
});

addLabel(ctx, slide, {
  x: 485,
  y: 413,
  w: 120,
  h: 18,
  text: "s_w回写量化网格",
  size: 9,
  color: "#2f5597",
  bold: true,
});
addLabel(ctx, slide, {
  x: 354,
  y: 499,
  w: 198,
  h: 16,
  text: "W * s_s改变权重分布 -> 触发s_w动态跟踪",
  size: 8.5,
  color: "#8a6200",
  bold: true,
});
addLabel(ctx, slide, {
  x: 606,
  y: 418,
  w: 54,
  h: 30,
  text: "协同\n耦合",
  size: 10,
  color: "#c00000",
  bold: true,
});

await fs.mkdir(preview.substring(0, preview.lastIndexOf("/")), { recursive: true }).catch(() => {});
const png = await presentation.export({ slide, format: "png", scale: 1 });
await saveBlobToFile(png, preview);

await fs.mkdir(output.substring(0, output.lastIndexOf("/")), { recursive: true }).catch(() => {});
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(output);
console.log(output);
