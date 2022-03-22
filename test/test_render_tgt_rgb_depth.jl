using PyCall, CUDA, Flux
using Monodepth
using Monodepth: get_src_xyz_from_plane_disparity

push!(pyimport("sys")."path", "D://Studie//Vop//MINE")
push!(pyimport("sys")."path", "/home/lab/Documents/vop_BC04/MINE")
push!(pyimport("sys")."path", "/mnt/c/Users/jules/OneDrive - UGent/Documenten/code/VOP/MINE/MINE")

@pyimport operations.mpi_rendering as mpi_rendering
@pyimport operations.homography_sampler as homography_sampler
@pyimport torch as torch
@pyimport numpy as np

cuda0 = torch.device("cuda:0")

B, S, H, W = 2, 32, 100, 200

mpi_rgb_src_p = rand(B,S,3,H,W)
mpi_rgb_src_j = permutedims(mpi_rgb_src_p, (5, 4, 3, 2, 1))|>gpu
mpi_sigma_src_p = rand(B,S,1,H,W)
mpi_sigma_src_j = permutedims(mpi_sigma_src_p, (5, 4, 3, 2, 1,))|>gpu
mpi_disparity_src_p = rand(B,S)
mpi_disparity_src_j = permutedims(mpi_disparity_src_p, (2, 1))|>gpu
xyz_tgt_BS3HW_p = rand(B,S,3,H,W)
xyz_tgt_BS3HW_j = permutedims(xyz_tgt_BS3HW_p, (5, 4, 3, 2, 1))|>gpu
K_src_inv_p = repeat(rand(1, 3, 3), B, 1, 1)
K_tgt_p = repeat(rand(1, 3, 3), B, 1, 1)

r = randn(3, B)
R = permutedims(so3_exp_map(r), (3, 1, 2))
t = randn(1, B, 3)

#G Julia
pose = Pose(gpu(r), permutedims(t[1, :, :], (2, 1))|>gpu)

#G python
G_tgt_src = zeros(1, B, 4, 4)
G_tgt_src[:, :, 1:3, 1:3] .= Flux.unsqueeze(R, 1)
G_tgt_src[:, :, 1:3, 4] .= t
G_tgt_src[:, :, 4, 4] .= 1
G_tgt_src = reshape(G_tgt_src, (B, 4, 4))
G_tgt_src = torch.tensor(G_tgt_src)

tgt_rgb_syn_p, tgt_depth_syn_p, tgt_mask_p = mpi_rendering.render_tgt_rgb_depth( ,mpi_rgb_src_p,mpi_sigma_src_p, mpi_disparity_src_p, xyz_tgt_BS3HW_p, G_tgt_src, K_src_inv_p, K_tgt_p )
tgt_rgb_syn_j, tgt_depth_syn_j, tgt_mask_j = mpi_rendering.render_tgt_rgb_depth( ,mpi_rgb_src_j,mpi_sigma_src_j, mpi_disparity_src_j, xyz_tgt_BS3HW_j, pose,K_src_inv_j[:, :, 1], K_tgt_p )
