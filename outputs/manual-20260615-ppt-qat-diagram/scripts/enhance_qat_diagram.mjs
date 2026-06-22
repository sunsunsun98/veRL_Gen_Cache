import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "file:///C:/Users/12708/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const sourcePptx = "D:/Program Files (x86)/code_repo/veRL/低比特专利ppt/QAT画图.pptx";
const finalPptx = sourcePptx;
const workspace = "D:/Program Files (x86)/code_repo/veRL/outputs/manual-20260615-ppt-qat-diagram";
const previewDir = path.join(workspace, "preview");
const layoutDir = path.join(workspace, "layout");

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addTextBox(slide, name, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: 15,
    color: "#334155",
    typeface: "Microsoft YaHei",
    ...style,
  };
  return shape;
}

function addRoundBox(slide, name, text, position, fill, lineFill, style = {}) {
  const shape = slide.shapes.add({
    geometry: "roundRect",
    name,
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: 1.5 },
    borderRadius: "rounded-lg",
  });
  shape.text = text;
  shape.text.style = {
    fontSize: 14,
    color: "#1f2937",
    typeface: "Microsoft YaHei",
    alignment: "center",
    ...style,
  };
  return shape;
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });

  const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePptx));
  const slide = presentation.resolve("sl/2x4nap4r");

  await fs.writeFile(path.join(layoutDir, "before-slide-01.layout.json"), await (await slide.export({ format: "layout" })).text());

  // Rename existing labels so the two learnable factors are visible at first glance.
  const smoothLabel = presentation.resolve("sh/25kb6xkf");
  smoothLabel.text = "learnable\nsmooth scale";
  smoothLabel.text.style = { fontSize: 14, bold: true, color: "#0f766e", typeface: "Microsoft YaHei", alignment: "center" };

  const weightMap = presentation.resolve("sh/q1orytsv");
  weightMap.text = "量化映射\nW / weight scale + zp";
  weightMap.text.style = { fontSize: 14, color: "#1f2937", typeface: "Microsoft YaHei", alignment: "center" };

  const weightDeq = presentation.resolve("sh/qtcj6tov");
  weightDeq.text = "反量化映射\n(Wint - zp) * weight scale";
  weightDeq.text.style = { fontSize: 14, color: "#1f2937", typeface: "Microsoft YaHei", alignment: "center" };

  const frameTitle = presentation.resolve("sh/lwval87e");
  frameTitle.text = "平滑感知QAT伪量化过程";
  frameTitle.text.style = { fontSize: 14, bold: true, color: "#1f2937", typeface: "Microsoft YaHei", alignment: "center" };

  // Non-covering row outlines emphasize the two branches without masking inherited objects.
  slide.shapes.add({
    geometry: "roundRect",
    name: "activation-smooth-branch-outline",
    position: { left: 92, top: 78, width: 1018, height: 145 },
    fill: "none",
    line: { style: "dashed", fill: "#14b8a6", width: 2 },
    borderRadius: "rounded-xl",
  });
  slide.shapes.add({
    geometry: "roundRect",
    name: "weight-scale-branch-outline",
    position: { left: 92, top: 232, width: 1018, height: 130 },
    fill: "none",
    line: { style: "dashed", fill: "#f97316", width: 2 },
    borderRadius: "rounded-xl",
  });

  addRoundBox(
    slide,
    "smooth-scale-role-card",
    "Smooth scale：激活侧平滑\nx' = x / s_smooth\n压缩离群值动态范围，降低A8截断/舍入扰动",
    { left: 338, top: 36, width: 360, height: 58 },
    "#ECFDF5",
    "#14b8a6",
    { fontSize: 13.5, color: "#0f513f" },
  );

  addRoundBox(
    slide,
    "weight-scale-role-card",
    "Weight scale：权重侧量化网格\nΔw随平滑后权重分布学习更新\n跟踪 W' = W × s_smooth 的动态范围",
    { left: 338, top: 354, width: 390, height: 64 },
    "#FFF7ED",
    "#f97316",
    { fontSize: 13.5, color: "#7c2d12" },
  );

  addRoundBox(
    slide,
    "dual-scale-core-card",
    "双尺度协同：smooth scale迁移激活异常值，weight scale重配权重量化步长\n前向保持线性层等价，反向通过STE近似梯度联合更新",
    { left: 742, top: 354, width: 376, height: 64 },
    "#EFF6FF",
    "#2563eb",
    { fontSize: 13.2, color: "#1e3a8a" },
  );

  addTextBox(
    slide,
    "activation-branch-tag",
    "激活路径：平滑后再伪量化",
    { left: 914, top: 94, width: 172, height: 25 },
    { fontSize: 13, bold: true, color: "#0f766e", alignment: "center" },
  );

  addTextBox(
    slide,
    "weight-branch-tag",
    "权重路径：量化因子可学习更新",
    { left: 890, top: 333, width: 206, height: 25 },
    { fontSize: 13, bold: true, color: "#c2410c", alignment: "center" },
  );

  // Use preset arrow shapes instead of connector lines; imported source connectors
  // do not have reliable endpoint metadata for rerouting in this deck.
  slide.shapes.add({
    geometry: "downArrow",
    name: "smooth-card-to-divide-arrow",
    position: { left: 216, top: 122, width: 26, height: 34 },
    fill: "#14b8a6",
    line: { style: "solid", fill: "#0f766e", width: 1 },
  });
  slide.shapes.add({
    geometry: "upArrow",
    name: "weight-card-to-scale-arrow",
    position: { left: 385, top: 323, width: 24, height: 30 },
    fill: "#f97316",
    line: { style: "solid", fill: "#c2410c", width: 1 },
  });
  slide.shapes.add({
    geometry: "downArrow",
    name: "smooth-weight-coupling-arrow",
    position: { left: 214, top: 198, width: 25, height: 74 },
    fill: "#2563eb",
    line: { style: "solid", fill: "#1d4ed8", width: 1 },
  });
  addTextBox(
    slide,
    "coupling-equivalence-label",
    "平滑重参数化\nx'/W'协同补偿",
    { left: 228, top: 207, width: 116, height: 44 },
    { fontSize: 12.5, color: "#1d4ed8", alignment: "center", bold: true },
  );

  await fs.writeFile(path.join(layoutDir, "after-slide-01.layout.json"), await (await slide.export({ format: "layout" })).text());

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);

  const verify = await presentation.inspect({
    kind: "slide,textbox,shape,layout",
    search: "smooth scale",
    maxChars: 12000,
  });
  await fs.writeFile(path.join(workspace, "inspect-after-smooth-scale.txt"), verify.ndjson);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
