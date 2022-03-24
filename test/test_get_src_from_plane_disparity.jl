using PyCall, CUDA, Flux
using Monodepth: get_src_xyz_from_plane_disparity

push!(pyimport("sys")."path", "D://Studie//Vop//MINE")
push!(pyimport("sys")."path", "/home/lab/Documents/vop_BC04/MINE")
push!(pyimport("sys")."path", "/mnt/c/Users/jules/OneDrive - UGent/Documenten/code/VOP/MINE/MINE")

@pyimport operations.mpi_rendering as mpi_rendering
@pyimport operations.homography_sampler as homography_sampler
@pyimport torch as torch
@pyimport numpy as np

B, S, H, W = 2, 32, 100, 200

cuda0 = torch.device("cuda:0")

# function get_src_xyz_from_plane_disparity(meshgrid_src_homo, mpi_disparity_src, K_src_inv)
#     N, B = size(mpi_disparity_src)

#     mpi_depth_src = reshape(1 ./ mpi_disparity_src, (1, 1, 1, N, B))
#     return reshape(K_src_inv * reshape(meshgrid_src_homo, 3, :), (3, W, H)) .* mpi_depth_src
# end
meshgrid_src_homo_p = rand(3,H,W)
mpi_disparity_src_p = rand(B,S)
K_src_inv_p = repeat(rand(1, 3, 3), B, 1, 1)

meshgrid_src_homo_j =  permutedims(meshgrid_src_homo_p, (1,3,2))|>gpu
mpi_disparity_src_j =  permutedims(mpi_disparity_src_p, (2,1))|>gpu
K_src_inv_j = permutedims(K_src_inv_p, (2,3,1))|>gpu

meshgrid_src_homo_p, mpi_disparity_src_p, K_src_inv_p = [torch.tensor(x, device=cuda0) for x in (meshgrid_src_homo_p, mpi_disparity_src_p, K_src_inv_p)]

out_p = mpi_rendering.get_src_xyz_from_plane_disparity(meshgrid_src_homo_p, mpi_disparity_src_p, K_src_inv_p)
out_j = get_src_xyz_from_plane_disparity(meshgrid_src_homo_j, mpi_disparity_src_j, K_src_inv_j[:, :, 1])
size(out_j)

isapprox(collect(permutedims(out_j, (5,4,1,3,2))), np.array(out_p.cpu()), rtol=1e-7)

collect(permutedims(out_j, (5,4,1,3,2))) .- np.array(out_p.cpu())


#pyton line by line
py"""



    
def get_src_xyz_from_plane_disparity(meshgrid_src_homo,
                                     mpi_disparity_src,
                                     K_src_inv):
 
    B, S = mpi_disparity_src.size()
    H, W = meshgrid_src_homo.size(1), meshgrid_src_homo.size(2)
    mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS

    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, 3, 3)

    # 3xHxW -> BxSx3xHxW
    meshgrid_src_homo = meshgrid_src_homo.unsqueeze(0).unsqueeze(1).repeat(B, S, 1, 1, 1)
    meshgrid_src_homo_Bs3N = meshgrid_src_homo.reshape(B * S, 3, -1)
    xyz_src = torch.matmul(K_src_inv_Bs33, meshgrid_src_homo_Bs3N)  # BSx3xHW
    xyz_src = xyz_src.reshape(B, S, 3, H * W) * mpi_depth_src.unsqueeze(2).unsqueeze(3)  # BxSx3xHW
    xyz_src_BS3HW = xyz_src.reshape(B, S, 3, H, W)

    return xyz_src_BS3HW
    """





