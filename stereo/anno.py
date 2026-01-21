import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class AnnPoint:
    label: str
    x: float
    y: float

def load_xanylabeling_points(json_path: str, label_filter: Optional[str] = None) -> List[AnnPoint]:
    """
    读取 X-AnyLabeling/LabelMe 风格 json，提取所有 point 标注。
    返回像素坐标 (x,y)（注意：x 是列，y 是行）
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts: List[AnnPoint] = []
    for sh in data.get("shapes", []):
        shape_type = sh.get("shape_type", "")
        label = sh.get("label", "")

        if shape_type != "point":
            continue
        if label_filter is not None and label != label_filter:
            continue

        p = sh.get("points", None)
        if not p:
            continue

        # 常见：[[x,y]]
        x, y = p[0]
        pts.append(AnnPoint(label=label, x=float(x), y=float(y)))

    return pts