#!/usr/bin/env python
"""Route A — fast object mesh directly from the VGGT point map (Open3D).

This is the "photogrammetry-style" route, but powered by VGGT's dense point
map instead of a classical SfM/MVS stack (no sudo / no COLMAP binary needed).
Best for: a clean, editable, watertight-ish mesh in seconds from few photos.

Pipeline: load fused point cloud -> statistical outlier removal -> estimate +
orient normals -> Poisson reconstruction -> trim low-density faces -> keep the
largest connected component (drops stray background blobs) -> export.

Usage:
    python routeA_photogrammetry/mesh_from_pointmap.py \
        --scene data/output/scene --out data/output/mesh_A.obj --depth 9
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/output/scene")
    ap.add_argument("--ply", default=None, help="override point cloud path (default <scene>/points.ply)")
    ap.add_argument("--out", default="data/output/mesh_A.obj")
    ap.add_argument("--depth", type=int, default=9, help="Poisson octree depth (8-10; higher = finer)")
    ap.add_argument("--density-quantile", type=float, default=0.05,
                    help="trim faces below this density quantile (removes Poisson balloons)")
    args = ap.parse_args()

    ply = args.ply or str(Path(args.scene) / "points.ply")
    pcd = o3d.io.read_point_cloud(ply)
    if len(pcd.points) == 0:
        raise SystemExit(f"Empty point cloud: {ply}")
    print(f"Loaded {len(pcd.points)} points from {ply}")

    # denoise
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # normals (required by Poisson)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)

    print(f"Poisson reconstruction (depth={args.depth}) ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth)
    densities = np.asarray(densities)
    keep = densities >= np.quantile(densities, args.density_quantile)
    mesh.remove_vertices_by_mask(~keep)

    # keep the largest connected component = the object
    mesh.remove_unreferenced_vertices()
    tri_clusters, n_tri, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    n_tri = np.asarray(n_tri)
    if len(n_tri):
        biggest = int(np.argmax(n_tri))
        mesh.remove_triangles_by_mask(tri_clusters != biggest)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"Mesh -> {args.out}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")


if __name__ == "__main__":
    main()
