"""Frozen PointNet++ shape vectors for the Weave object meshes.

Each object becomes a 1024-d unit vector from the ModelNet40 PointNet++ in
``torch-pointcloud``, plus the mesh extents in meters. The vector is computed
once per mesh, in the object's own frame, and reused for every clip of that
object. Neighbor search in that checkpoint normally calls ``pyg-lib``, which
is not installed here, so those two ops are a small PyTorch stand-in used only
while encoding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

ENCODER_NAME = "pointnet2.modelnet40.openpoints"
POINT_COUNT = 1024
SHAPE_DIM = 1027
_CACHE = Path(__file__).resolve().parents[3] / "assets" / "object_geom.npz"


def _pytorch_fps(x, batch=None, ratio=0.5, random_start=True, batch_size=None):
    """Farthest-point indices. ``ratio`` is the fraction kept in each cloud."""
    del batch_size
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    chosen = []
    count = int(batch.max().item()) + 1 if batch.numel() else 0
    for item in range(count):
        points = x[batch == item]
        offset = int((batch < item).sum().item())
        number = points.shape[0]
        keep = max(1, int(np.ceil(float(ratio) * number)))
        keep = min(keep, number)
        start = int(torch.randint(number, ()).item()) if random_start else 0
        selected = torch.empty(keep, dtype=torch.long, device=x.device)
        selected[0] = start
        nearest = torch.full((number,), float("inf"), device=x.device)
        for step in range(1, keep):
            last = points[selected[step - 1]]
            nearest = torch.minimum(nearest, (points - last).square().sum(-1))
            nearest[selected[:step]] = -1.0
            selected[step] = int(torch.argmax(nearest).item())
        chosen.append(selected + offset)
    if not chosen:
        return torch.zeros(0, dtype=torch.long, device=x.device)
    return torch.cat(chosen)


def _pytorch_radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32, num_workers=1, batch_size=None):
    """Ball-query edges ``(2, E)``: row is the query, column is the source.

    Inside each ball the smallest source indices are kept, matching the
    PointNet++ reference kernel.
    """
    del num_workers, batch_size
    device = x.device
    if batch_x is None:
        batch_x = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    if batch_y is None:
        batch_y = torch.zeros(y.shape[0], dtype=torch.long, device=device)
    limit = int(max_num_neighbors)
    radius_sq = float(r) * float(r)
    rows = []
    cols = []
    batches = int(max(int(batch_x.max().item()) if batch_x.numel() else -1, int(batch_y.max().item()) if batch_y.numel() else -1)) + 1
    for item in range(batches):
        source = x[batch_x == item]
        query = y[batch_y == item]
        if source.numel() == 0 or query.numel() == 0:
            continue
        source_index = torch.nonzero(batch_x == item, as_tuple=False).flatten()
        query_index = torch.nonzero(batch_y == item, as_tuple=False).flatten()
        distance = (query.unsqueeze(1) - source.unsqueeze(0)).square().sum(-1)
        local = torch.arange(source.shape[0], device=device).view(1, -1).expand(query.shape[0], -1).clone()
        local = local.masked_fill(distance > radius_sq, source.shape[0])
        picked = torch.sort(local, dim=-1).values[:, :limit]
        valid = picked < source.shape[0]
        rows.append(query_index[torch.arange(query.shape[0], device=device).unsqueeze(1).expand_as(picked)[valid]])
        cols.append(source_index[picked[valid]])
    if not rows:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return torch.stack((empty, empty))
    return torch.stack((torch.cat(rows), torch.cat(cols)))


def _install_neighbor_ops() -> None:
    import torch_geometric.nn.pool as pool

    pool.fps = _pytorch_fps
    pool.radius = _pytorch_radius


def _mesh_points(path: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a triangle mesh.")
    extents = (loaded.vertices.max(axis=0) - loaded.vertices.min(axis=0)).astype(np.float32)
    seed = int(np.frombuffer(path.stem.encode(), dtype=np.uint8).sum())
    points, _ = trimesh.sample.sample_surface(loaded, count, seed=seed)
    points = np.asarray(points, dtype=np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(points, axis=1).max())
    points = points / max(radius, 1e-8)
    return points, extents


def _encode_points(points: np.ndarray) -> np.ndarray:
    import torch_pointcloud as tp

    _install_neighbor_ops()
    model = tp.create_model(ENCODER_NAME, pretrained=True).eval()
    model.reset_classifier(num_classes=0)
    pos = torch.from_numpy(np.ascontiguousarray(points))
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    with torch.no_grad():
        embedding = model(pos, pos, batch)[0]
    embedding = embedding / embedding.norm().clamp_min(1e-8)
    return embedding.cpu().numpy().astype(np.float32)


def _load_cache() -> dict[str, np.ndarray]:
    if not _CACHE.is_file():
        return {}
    stored = np.load(_CACHE, allow_pickle=False)
    if int(stored["points"]) != POINT_COUNT or str(stored["encoder"]) != ENCODER_NAME:
        return {}
    return {str(name): vector for name, vector in zip(stored["names"], stored["vectors"], strict=True)}


def _write_cache(vectors: dict[str, np.ndarray]) -> None:
    names = np.asarray(sorted(vectors))
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        _CACHE,
        names=names,
        vectors=np.stack([vectors[name] for name in names]),
        points=np.int64(POINT_COUNT),
        encoder=np.asarray(ENCODER_NAME),
    )


def object_shape_table(root: Path, names: np.ndarray) -> torch.Tensor:
    """One shape vector per clip, aligned with ``names``. Shape ``(clips, 1027)``."""
    requested = [str(name) for name in np.asarray(names).tolist()]
    cached = _load_cache()
    missing = sorted({name for name in requested if name not in cached})
    if missing:
        print(f"encoding object shapes: {', '.join(missing)}", flush=True)
        for name in missing:
            path = root / "objects" / name / f"{name}.obj"
            if not path.is_file():
                raise FileNotFoundError(path)
            points, extents = _mesh_points(path, POINT_COUNT)
            cached[name] = np.concatenate((_encode_points(points), extents))
        _write_cache(cached)
    rows = np.stack([cached[name] for name in requested]).astype(np.float32)
    if rows.shape[1] != SHAPE_DIM:
        raise ValueError(f"Expected shape dim {SHAPE_DIM}, got {rows.shape[1]}.")
    return torch.from_numpy(rows)
