using PyCall, CUDA, Flux
using Monodepth
using Monodepth: get_src_xyz_from_plane_disparity,so3_exp_map, Pose, render_tgt_rgb_depth

push!(pyimport("sys")."path", "D://Studie//Vop//MINE")
push!(pyimport("sys")."path", "/home/lab/Documents/vop_BC04/MINE")
push!(pyimport("sys")."path", "/mnt/c/Users/jules/OneDrive - UGent/Documenten/code/VOP/MINE/MINE")

@pyimport operations.mpi_rendering as mpi_rendering
@pyimport operations.homography_sampler as homography_sampler
@pyimport torch as torch
@pyimport numpy as np

cuda0 = torch.device("cuda:0")

B, N, H, W = 2, 32, 100, 200

mpi_rgb_src_p = rand(B,N,3,H,W)
mpi_rgb_src_j = permutedims(mpi_rgb_src_p, (5, 4, 3, 2, 1))|>gpu
# mpi_rgb_src_p = torch.tensor(mpi_rgb_src_p, device=cuda0)
mpi_sigma_src_p = rand(B,N,1,H,W)
mpi_sigma_src_j = permutedims(mpi_sigma_src_p, (5, 4, 3, 2, 1,))|>gpu
# mpi_sigma_src_p = torch.tensor(mpi_sigma_src_p, device=cuda0)
mpi_disparity_src_p = rand(B,N)
mpi_disparity_src_j = permutedims(mpi_disparity_src_p, (2, 1))|>gpu
# mpi_disparity_src_p = torch.tensor(mpi_disparity_src_p, device=cuda0).contiguous()

xyz_tgt_BS3HW_p = rand(B,N,3,H,W)
# xyz_tgt_BS3HW_j = permutedims(xyz_tgt_BS3HW_p, (5, 4, 3, 2, 1))|>gpu
xyz_tgt_BS3HW_j = permutedims(xyz_tgt_BS3HW_p, (3, 5, 4, 2, 1))|>gpu
# xyz_tgt_BS3HW_p = torch.tensor(xyz_tgt_BS3HW_p, device=cuda0)
K_src_inv_p = repeat(rand(1, 3, 3), B, 1, 1)
K_tgt_p = repeat(rand(1, 3, 3), B, 1, 1)

K_src_inv_j = K_src_inv_p[1, :, :]
K_tgt_j = K_tgt_p[1,:,:]

#K_src_inv_p = torch.tensor(K_src_inv_p, device=cuda0)
#K_tgt_p = torch.tensor(K_tgt_p, device=cuda0)


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
# G_t_src, device=cudatgt_src = torch.tensor(G_tg0)
mpi_rgb_src_p, mpi_sigma_src_p, mpi_disparity_src_p, xyz_tgt_BS3HW_p,K_src_inv_p,K_tgt_p, G_tgt_src = [torch.tensor(Float32.(x), device=cuda0).contiguous() for x in (mpi_rgb_src_p, mpi_sigma_src_p, mpi_disparity_src_p, xyz_tgt_BS3HW_p,K_src_inv_p,K_tgt_p, G_tgt_src)]

HS = homography_sampler.HomographySample(H, W, device=cuda0)

tgt_rgb_syn_p, tgt_depth_syn_p, tgt_mask_p = mpi_rendering.render_tgt_rgb_depth( HS ,mpi_rgb_src_p,mpi_sigma_src_p, mpi_disparity_src_p, xyz_tgt_BS3HW_p, G_tgt_src, K_src_inv_p, K_tgt_p )
tgt_rgb_syn_j, tgt_depth_syn_j, tgt_mask_j = render_tgt_rgb_depth(gpu(mpi_rgb_src_j),gpu(mpi_sigma_src_j), gpu(mpi_disparity_src_j), gpu(xyz_tgt_BS3HW_j), gpu(pose),gpu(K_src_inv_j[:, :, 1]), gpu(K_tgt_j ))

