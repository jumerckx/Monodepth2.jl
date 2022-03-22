using Monodepth: get_tgt_xyz_from_plane_disparity, so3_exp_map, Pose
using PyCall, CUDA, Flux

# push!(pyimport("sys")."path", "D://Studie//Vop//MINE")
push!(pyimport("sys")."path", "/mnt/c/Users/jules/OneDrive - UGent/Documenten/code/VOP/MINE/MINE")


@pyimport operations.mpi_rendering as mpi_rendering
@pyimport torch as torch
@pyimport numpy as np

B, N, H, W = 4, 32, 100, 200

cuda0 = torch.device("cuda:0")

#xyz python en julia
xyz_BS3HW = rand(B, N, 3, H, W)
xyz = permutedims(xyz_BS3HW, (3, 5, 4, 2, 1))|>gpu
xyz_BS3HW = torch.tensor(xyz_BS3HW)

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

xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(xyz_BS3HW, G_tgt_src)
xyz_tgt_BS3HW2 = get_tgt_xyz_from_plane_disparity(gpu(xyz), gpu(pose))

isapprox(collect(permutedims(xyz_tgt_BS3HW2, (5, 4, 1, 3, 2))), np.array(xyz_tgt_BS3HW.cpu()), rtol=1e-7)