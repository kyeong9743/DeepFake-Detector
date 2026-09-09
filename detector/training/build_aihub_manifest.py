"""
AI Hub 「딥페이크 변조 영상」(003.딥페이크) -> CAKE manifest.csv 생성.

데이터셋 구조 (마운트 /aihub 기준):
  003.딥페이크/1.Training/라벨링데이터/train_meta_data/{원본,변조}영상_training_메타데이터.csv
  003.딥페이크/1.Training/원천데이터/train_원본/원본N/{UUID}/{UUID}_NNN.mp4            (label 0, HEVC ~90s)
  003.딥페이크/1.Training/원천데이터/train_변조/{method}N/{UUID}/{ID}.mp4               (label 1, H264 ~20s)
  003.딥페이크/1.Training/원천데이터/train_탐지방해/{원본영상|변조영상}/NN_*/*.mp4       (탐지 방해 처리본)
  003.딥페이크/2.Validation/...  동일 구조 (validate_*)

라벨: 폴더 위치로 확정, 메타 CSV 로 변조모델·UUID(인물)·타겟 보강.
그룹: real = UUID(인물), fake = 타겟UUID(변조된 인물). 같은 인물의 영상이 train/val 에 갈라지지 않게 함(누출 방지).
샘플링: 전체 20만 개 중 --per-class 개씩 균형 추출. fake 는 변조 방법별 균등, 타겟영상당 최대 --max-per-target 개.

    python -m training.build_aihub_manifest --aihub /aihub --per-class 3000 --out /data/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("aihub_manifest")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
_METHOD_RE = re.compile(r"^([a-z_]+?)\d*$") # fo1 -> fo, audio_driven2 -> audio_driven


def load_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def scan(root: Path) -> dict[str, Path]:
    """root 아래 모든 mp4 -> {파일명: 경로}. 파일명은 데이터셋 전체에서 고유."""
    out: dict[str, Path] = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if Path(fn).suffix.lower() in VIDEO_EXT:
                out[fn] = Path(dp) / fn
    return out


def method_of(path: Path, src_root: Path) -> str:
    """train_변조/fo3/UUID/x.mp4 -> fo"""
    rel = path.relative_to(src_root).parts
    m = _METHOD_RE.match(rel[0]) if rel else None
    return m.group(1) if m else "unknown"


def build_split(aihub: Path, split_name: str, rng: random.Random, per_class: int, max_per_target: int,
                include_anti: bool, methods: set[str] | None) -> list[dict]:
    top = aihub / "003.딥페이크" / ("1.Training" if split_name == "train" else "2.Validation")
    tag = "train" if split_name == "train" else "validate"
    meta_dir = top / "라벨링데이터" / f"{tag}_meta_data"
    src = top / "원천데이터"
    if not src.exists():
        raise FileNotFoundError(f"원천데이터 없음: {src}")

    real_meta = {r["영상ID"]: r for r in load_csv(next(meta_dir.glob("원본영상_*.csv")))}
    fake_meta = {r["영상ID"]: r for r in load_csv(next(meta_dir.glob("변조영상_*.csv")))}
    logger.info("[%s] 메타: 원본 %d, 변조 %d", split_name, len(real_meta), len(fake_meta))

    real_files = scan(src / f"{tag}_원본")
    fake_files = scan(src / f"{tag}_변조")
    logger.info("[%s] 디스크: 원본 %d, 변조 %d", split_name, len(real_files), len(fake_files))

    # ---- real: 인물(UUID) 단위로 골고루 ----
    by_person: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for fn, p in real_files.items():
        uuid = real_meta.get(fn, {}).get("UUID") or fn.split("_")[0]
        by_person[uuid].append((fn, p))
    persons = sorted(by_person); rng.shuffle(persons)
    reals: list[dict] = []
    # round-robin 으로 인물당 1개씩 돌며 per_class 채움 -> 인물 다양성 최대
    idx = {u: 0 for u in persons}
    while len(reals) < per_class and any(idx[u] < len(by_person[u]) for u in persons):
        for u in persons:
            if idx[u] < len(by_person[u]) and len(reals) < per_class:
                fn, p = sorted(by_person[u])[idx[u]]; idx[u] += 1
                m = real_meta.get(fn, {})
                reals.append({"path": str(p), "label": 0, "source": "aihub_real", "method": "real",
                              "group": u, "gender": m.get("인물성별", ""), "split": split_name})

    # ---- fake: 변조 방법별 균등, 타겟영상당 max_per_target ----
    by_method: dict[str, dict[str, list[tuple[str, Path]]]] = defaultdict(lambda: defaultdict(list))
    for fn, p in fake_files.items():
        m = fake_meta.get(fn, {})
        method = m.get("변조모델") or method_of(p, src / f"{tag}_변조")
        if methods and method not in methods:
            continue
        target = m.get("타겟영상") or fn
        by_method[method][target].append((fn, p))
    fakes: list[dict] = []
    mlist = sorted(by_method)
    quota = {m: per_class // len(mlist) + (1 if i < per_class % len(mlist) else 0) for i, m in enumerate(mlist)}
    for method in mlist:
        targets = sorted(by_method[method]); rng.shuffle(targets)
        picked = 0; ti = {t: 0 for t in targets}
        while picked < quota[method] and any(ti[t] < min(max_per_target, len(by_method[method][t])) for t in targets):
            for t in targets:
                if picked >= quota[method]:
                    break
                lst = sorted(by_method[method][t])
                if ti[t] < min(max_per_target, len(lst)):
                    fn, p = lst[ti[t]]; ti[t] += 1; picked += 1
                    m = fake_meta.get(fn, {})
                    fakes.append({"path": str(p), "label": 1, "source": "aihub_fake", "method": method,
                                  "group": m.get("타겟UUID") or t.split("_")[0], "gender": m.get("인물성별", ""),
                                  "split": split_name})
        logger.info("[%s] 변조 %-13s 타겟 %5d개 중 %5d 선택 (할당 %d)", split_name, method, len(targets), picked, quota[method])

    # ---- 탐지방해 (선택): 원본영상/변조영상 하위 폴더명으로 라벨 ----
    antis: list[dict] = []
    if include_anti:
        aroot = src / f"{tag}_탐지방해"
        for sub, label in (("원본영상", 0), ("변조영상", 1)):
            files = scan(aroot / sub)
            names = sorted(files); rng.shuffle(names)
            n = max(1, per_class // 10) # 클래스당 10% 를 방해 처리본으로
            for fn in names[:n]:
                grp = fn.split("_")[0]
                antis.append({"path": str(files[fn]), "label": label, "source": "aihub_anti", "method": f"anti_{'real' if label==0 else 'fake'}",
                              "group": grp, "gender": "", "split": split_name})
        logger.info("[%s] 탐지방해 %d 추가", split_name, len(antis))

    rows = reals + fakes + antis
    logger.info("[%s] 합계 %d (real %d / fake %d)", split_name, len(rows), sum(r["label"] == 0 for r in rows), sum(r["label"] == 1 for r in rows))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Hub 딥페이크 데이터셋 -> manifest.csv")
    ap.add_argument("--aihub", type=Path, default=Path("/aihub"), help="컨테이너 안의 AI Hub 데이터 경로 (003.딥페이크 의 상위, compose 오버라이드가 /aihub 로 마운트)")
    ap.add_argument("--out", type=Path, default=Path("/data/manifest.csv"))
    ap.add_argument("--per-class", type=int, default=3000, help="train 클래스별 영상 수")
    ap.add_argument("--val-per-class", type=int, default=600, help="validation 클래스별 영상 수 (0=미포함)")
    ap.add_argument("--max-per-target", type=int, default=2, help="같은 타겟(원본) 영상에서 뽑을 변조본 최대 수")
    ap.add_argument("--methods", default="", help="쉼표 구분 변조 방법 필터 (예: fo,dfl,fsgan). 비우면 전부")
    ap.add_argument("--include-anti", action="store_true", help="탐지방해 처리본 포함 (클래스당 10%%)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = random.Random(args.seed)
    methods = {m.strip() for m in args.methods.split(",") if m.strip()} or None

    rows = build_split(args.aihub, "train", rng, args.per_class, args.max_per_target, args.include_anti, methods)
    if args.val_per_class > 0:
        rows += build_split(args.aihub, "val", rng, args.val_per_class, args.max_per_target, args.include_anti, methods)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "method", "group", "gender", "split"])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    logger.info("저장: %s (%d행)", args.out, len(rows))
    logger.info("method 분포: %s", dict(Counter(r["method"] for r in rows)))
    logger.info("split 분포: %s", dict(Counter((r["split"], r["label"]) for r in rows)))
    logger.info("인물(group) 수: %d", len({r["group"] for r in rows}))


if __name__ == "__main__":
    main()
