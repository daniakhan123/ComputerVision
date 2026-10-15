import os
import subprocess
import open3d as o3d

PROJECT = "colmap_project"
IMAGES = "images"
DB = PROJECT + "/database.db"
SPARSE = PROJECT + "/sparse"
DENSE = PROJECT + "/dense"


def run(command):
    print(" ".join(command))
    subprocess.run(command, check=True)


def sparse_reconstruction():
    os.makedirs(SPARSE, exist_ok=True)
    run(["colmap", "feature_extractor", "--database_path", DB, "--image_path", IMAGES,
         "--ImageReader.single_camera", "1"])
    run(["colmap", "exhaustive_matcher", "--database_path", DB])
    run(["colmap", "mapper", "--database_path", DB, "--image_path", IMAGES, "--output_path", SPARSE])
    run(["colmap", "model_converter", "--input_path", SPARSE + "/0",
         "--output_path", PROJECT + "/sparse.ply", "--output_type", "PLY"])


def dense_reconstruction():
    os.makedirs(DENSE, exist_ok=True)
    run(["colmap", "image_undistorter", "--image_path", IMAGES, "--input_path", SPARSE + "/0",
         "--output_path", DENSE, "--output_type", "COLMAP"])
    run(["colmap", "patch_match_stereo", "--workspace_path", DENSE])
    run(["colmap", "stereo_fusion", "--workspace_path", DENSE, "--output_path", DENSE + "/fused.ply"])
    run(["colmap", "poisson_mesher", "--input_path", DENSE + "/fused.ply",
         "--output_path", DENSE + "/mesh.ply"])


def mesh_from_sparse():
    cloud = o3d.io.read_point_cloud(PROJECT + "/sparse.ply")
    cloud.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=8)
    path = PROJECT + "/mesh_from_sparse.ply"
    o3d.io.write_triangle_mesh(path, mesh)
    return path


def view(cloud_path, mesh_path):
    cloud = o3d.io.read_point_cloud(cloud_path)
    print("Points in cloud:", len(cloud.points))
    o3d.visualization.draw_geometries([cloud], window_name="Point cloud")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name="Mesh")


os.makedirs(PROJECT, exist_ok=True)
sparse_reconstruction()

try:
    dense_reconstruction()
    view(DENSE + "/fused.ply", DENSE + "/mesh.ply")
except subprocess.CalledProcessError:
    print("Dense step failed (probably no NVIDIA GPU). Building the mesh from the sparse point cloud.")
    view(PROJECT + "/sparse.ply", mesh_from_sparse())
