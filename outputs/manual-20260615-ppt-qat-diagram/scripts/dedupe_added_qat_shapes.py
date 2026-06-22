import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


TARGET_NAMES = {
    "activation-smooth-branch-outline",
    "weight-scale-branch-outline",
    "smooth-scale-role-card",
    "weight-scale-role-card",
    "dual-scale-core-card",
    "activation-branch-tag",
    "weight-branch-tag",
    "smooth-card-to-divide-arrow",
    "weight-card-to-scale-arrow",
    "smooth-weight-coupling-arrow",
    "coupling-equivalence-label",
}

NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
ET.register_namespace("p", NS["p"])
ET.register_namespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")


def shape_name(sp):
    c_nv_pr = sp.find(".//p:cNvPr", NS)
    if c_nv_pr is None:
        return ""
    return c_nv_pr.attrib.get("name", "")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: dedupe_added_qat_shapes.py <input.pptx> <output.pptx>")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_path, "r") as zin:
        slide_xml = zin.read("ppt/slides/slide1.xml")
        root = ET.fromstring(slide_xml)
        sp_tree = root.find(".//p:cSld/p:spTree", NS)
        if sp_tree is None:
            raise RuntimeError("slide1.xml does not contain p:spTree")

        seen = set()
        removed = 0
        for child in list(sp_tree):
            if child.tag != f"{{{NS['p']}}}sp":
                continue
            name = shape_name(child)
            if name not in TARGET_NAMES:
                continue
            if name in seen:
                sp_tree.remove(child)
                removed += 1
            else:
                seen.add(name)

        new_slide_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "ppt/slides/slide1.xml":
                    data = new_slide_xml
                zout.writestr(item, data)

    print(f"removed={removed}")


if __name__ == "__main__":
    main()
