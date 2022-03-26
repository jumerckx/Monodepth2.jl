using Monodepth: sample, so3_exp_map, Pose
using PyCall, CUDA, Flux
push!(pyimport("sys")."path", "/mnt/c/Users/jules/OneDrive - UGent/Documenten/code/VOP/MINE/MINE")

@pyimport operations.homography_sampler as homography_sampler
@pyimport torch as torch
@pyimport numpy as np

cuda0 = torch.device("cuda:0")

B, N, H, W = 2, 32, 100, 200
HS = homography_sampler.HomographySample(H, W, device=cuda0)

# genereren van dummy-data:
begin

    src_BCHW = rand(B*N, 7, H, W)
    d_src_B = rand(B*N)

    r = randn(3, B)
    R = permutedims(so3_exp_map(r), (3, 1, 2))
    t = randn(1, B, 3)

    # G_tgt_src = zeros(B, N, 4, 4)
    # G_tgt_src[:, :, 1:3, 1:3] .= Flux.unsqueeze(R, 2)
    # G_tgt_src[:, :, 1:3, 4] .= t
    # G_tgt_src[:, :, 4, 4] .= 1
    # G_tgt_src = reshape(G_tgt_src, (B*N, 4, 4))

    G_tgt_src = zeros(N, B, 4, 4)
    G_tgt_src[:, :, 1:3, 1:3] .= Flux.unsqueeze(R, 1)
    G_tgt_src[:, :, 1:3, 4] .= t
    G_tgt_src[:, :, 4, 4] .= 1
    G_tgt_src = reshape(G_tgt_src, (B*N, 4, 4))

    # Julia:
    K_src_inv = repeat(rand(1, 3,3), B*N, 1, 1)
    K_tgt = repeat(K_src_inv[1:1, :, :], B*N, 1, 1)

    src = permutedims(src_BCHW, (4,3,2,1))|>gpu
    depth_src = reshape(d_src_B, N, B)|>gpu
    pose = Pose(gpu(r), permutedims(t[1, :, :], (2, 1))|>gpu)
    K = K_tgt[1, :, :]|>gpu
    K_inv = K_src_inv[1, :, :]|>gpu

    # Pytorch, to gpu:
    src_BCHW, d_src_B, G_tgt_src, K_src_inv, K_tgt = [torch.tensor(Float32.(x), device=cuda0) for x in (src_BCHW, d_src_B, G_tgt_src, K_src_inv, K_tgt)]
end

tgt_BCHW2, valid_mask2 = sample(gpu(src), gpu(depth_src), gpu(pose), gpu(K), gpu(K_inv))

src_BCHW, d_src_B, G_tgt_src, K_src_inv, K_tgt = [torch.tensor(Float32.(x), device=cuda0) for x in (src_BCHW, d_src_B, G_tgt_src, K_src_inv, K_tgt)]
tgt_BCHW, valid_mask = HS.sample(src_BCHW, d_src_B, G_tgt_src, K_src_inv, K_tgt)

# gebruik functies `j` (julia) en `p` (python) om te vermijden dat tussenvariabelen in globale scope belanden,
# met een return op gepaste lijn in onderstaande functies kan een variabele naar buiten gebracht worden door j() en p().

j = () -> begin
    W, H, _, _ = size(src)
    R = Monodepth.so3_exp_map(pose.rvec)
    t = Flux.unsqueeze(pose.tvec, 2)
    n = Monodepth.transfer([0 0 1])
    temp = Flux.unsqueeze(t ⊠ n, 3) ./ -reshape(depth_src, (1, 1, size(depth_src, 1), size(depth_src, 2)))
    # @show size(te mp), size(R)
    H_tgt_src = K ⊠ (reshape(Flux.unsqueeze(R, 3) .- temp, (3, 3, :))) ⊠ K_inv
    H_src_tgt = Monodepth.inv(H_tgt_src)

    meshgrid_tgt_homo = Monodepth.transfer(reshape(Monodepth.create_meshgrid(H, W), (3, :)) .- [1, 1, 0]) # TODO: misschien beter cachen?
    meshgrid_src = H_src_tgt ⊠ meshgrid_tgt_homo
    
    return meshgrid_src
    meshgrid_src = meshgrid_src[1:2, :, :] ./ meshgrid_src[3:3, :, :]
    return meshgrid_src
    valid_mask = (meshgrid_src[1, :, :] .< W .* meshgrid_src[1, :, :] .> -1) .* (meshgrid_src[2, :, :] .< H .* meshgrid_src[2, :, :] .> -1)
    return valid_mask
    meshgrid_src[1, :, :] .= (meshgrid_src[1, :, :] .+ eltype(meshgrid_src)(0.5)) ./ (W/2)
    meshgrid_src[2, :, :] .= (meshgrid_src[2, :, :] .+ eltype(meshgrid_src)(0.5)) ./ (H/2)
    
    meshgrid_src = reshape(meshgrid_src, (2, W, H, :))

    tgt = grid_sample(src, meshgrid_src; padding_mode=:border)
    return tgt, valid_mask
end

p = () -> begin
    B, channels, Height_src, Width_src = src_BCHW.size(0), src_BCHW.size(1), src_BCHW.size(2), src_BCHW.size(3)
    R_tgt_src = py"$(G_tgt_src)[:, 0:3, 0:3]"
    t_tgt_src = py"$(G_tgt_src)[:, 0:3, 3]"

    Height_tgt = HS.Height_tgt
    Width_tgt = HS.Width_tgt

    n = HS.n.to(device=src_BCHW.device)
    n = n.unsqueeze(0).repeat(B, 1)  # Bx3
    # Bx3x3 - (Bx3x1 * Bx1x3)
    # note here we use -d_src, because the plane function is n^T * X - d_src = 0
    d_src_B33 = d_src_B.reshape(B, 1, 1).repeat(1, 3, 3)  # B -> Bx3x3
    R_tnd = R_tgt_src - torch.matmul(t_tgt_src.unsqueeze(2), n.unsqueeze(1)) / -d_src_B33
    H_tgt_src = torch.matmul(K_tgt, torch.matmul(R_tnd, K_src_inv))
    py"""
    import torch
    from utils import inverse
    with torch.no_grad():
                H_src_tgt = inverse($H_tgt_src)
    """
    H_src_tgt = py"H_src_tgt"
    meshgrid_tgt_homo = HS.meshgrid.to(src_BCHW.device)
    # 3xHxW -> Bx3xHxW
    meshgrid_tgt_homo = meshgrid_tgt_homo.unsqueeze(0).expand(B, 3, Height_tgt, Width_tgt)

    # wrap meshgrid_tgt_homo to meshgrid_src
    meshgrid_tgt_homo_B3N = meshgrid_tgt_homo.view(B, 3, -1)  # Bx3xHW
    meshgrid_src_homo_B3N = torch.matmul(H_src_tgt, meshgrid_tgt_homo_B3N)  # Bx3x3 * Bx3xHW -> Bx3xHW
    # Bx3xHW -> Bx3xHxW -> BxHxWx3
    meshgrid_src_homo = meshgrid_src_homo_B3N.view(B, 3, Height_tgt, Width_tgt).permute(0, 2, 3, 1)
    return meshgrid_src_homo
    meshgrid_src = py"$meshgrid_src_homo[:, :, :, 0:2] / $meshgrid_src_homo[:, :, :, 2:]"  # BxHxWx2
    return meshgrid_src
    valid_mask_x = py"torch.logical_and($meshgrid_src[:, :, :, 0] < $Width_src, $meshgrid_src[:, :, :, 0] > -1)"
    valid_mask_y = py"torch.logical_and($meshgrid_src[:, :, :, 1] < $Height_src, $meshgrid_src[:, :, :, 1] > -1)"
    valid_mask = torch.logical_and(valid_mask_x, valid_mask_y)  # BxHxW
    return valid_mask
    # sample from src_BCHW
    # normalize meshgrid_src to [-1,1]

    py"""
    meshgrid_src = $meshgrid_src
    meshgrid_src[:, :, :, 0] = (meshgrid_src[:, :, :, 0]+0.5) / ($Width_src * 0.5) - 1
    meshgrid_src[:, :, :, 1] = (meshgrid_src[:, :, :, 1]+0.5) / ($Height_src * 0.5) - 1
    """
    meshgrid_src = py"meshgrid_src"

    torch.nn.functional.grid_sample(src_BCHW, grid=meshgrid_src, padding_mode="border", align_corners=false)
end

j()

j()[1:2, :, :] ./ j()[3:3, :, :]

reshape(j(),(3, W, H, :)) 

p().shape
py"$(p())[:, :, :, 2:]".shape

py"$(p())[:, :, :, 0:2] / $(p())[:, :, :, 2:]".shape
p()

permutedims(reshape(j(),(3, W, H, :)) , (4, 3, 2, 1))|>collect ≈ np.array(p().cpu())

isapprox((permutedims(reshape(j()[1:2, :, :] ./ j()[3:3, :, :],(2, W, H, :)) , (4, 3, 2, 1))|>collect), np.array(py"$(p())[:, :, :, 0:2] / $(p())[:, :, :, 2:]".cpu()), rtol=1)

test = (permutedims(reshape(j()[1:2, :, :] ./ j()[3:3, :, :],(2, W, H, :)) , (4, 3, 2, 1))|>collect) .- np.array(py"$(p())[:, :, :, 0:2] / $(p())[:, :, :, 2:]".cpu())


test2 = abs.((permutedims(reshape(j(),(3, W, H, :)) , (4, 3, 2, 1))|>collect) .- np.array(p().cpu()))

maximum(abs.(test))
np.array(p().cpu())[abs.(test2) .> 5]