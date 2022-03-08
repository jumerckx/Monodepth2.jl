W, H = 416, 128
N, B = 32, 4

invK = CUDA.rand(3, 3)


function create_meshgrid(H, W)
    cat((1:W) .* ones(H)', (1:H)' .* ones(W), ones(W, H), dims=3) # TODO: offset van 0.5px?
end

mpi_disparity_src = uniformly_sample_disparity_from_linspace_bins(N, B)

meshgrid_src_homo = transfer(create_meshgrid(H, W))

function get_src_xyz_from_plane_disparity(meshgrid_src_homo, mpi_disparity_src, K_src_inv)
    N, B = size(mpi_disparity_src)

    mpi_depth_src = reshape(1 ./ mpi_disparity_src, (1, 1, 1, N, B))

    return permutedims(
        reshape(
            K_src_inv * reshape(permutedims(meshgrid_src_homo, (3, 1, 2)), 3, :), (3,W, H)
            ), (2, 3, 1)) .* mpi_depth_src
end

using LinearAlgebra
import LinearAlgebra:norm
norm(x; dims=:) = sqrt.(sum(abs2.(x), dims=dims))

CUDA.@time xyz_src = get_src_xyz_from_plane_disparity(meshgrid_src_homo, mpi_disparity_src, invK)

function plane_volume_rendering(rgb, sigma, xyz)
    diff = xyz[:, :, :, 2:end,:] .- xyz[:, :, :, 1:end-1, :]
    dist = cat(norm(diff, dims=3), CUDA.fill(1f3, W, H, 1, 1, B), dims=4) # TODO: TinyNERF gebruikt fill met 1e10

    transparency = exp.(-dist .* sigma)
    alpha = 1 .- transparency

    transparency_acc = cumprod(transparency .+ 1e-6, dims=4) # TODO: is ".+ 1e-6 " nodig?
    transparency_acc = cat(CUDA.ones(W, H, 1, 1, B), transparency_acc[:, :, :, 1:end-1, :], dims=4)

    weights = transparency_acc .* alpha  # BxSx1xHxW

    rgb_out = dropdims(sum(sum(weights, dims=4) .* rgb, dims=4), dims=4)

    # TODO: return depth?
    return rgb_out, transparency_acc, weights
end

rgb, sigma = CUDA.rand(W, H, 3, N, B), CUDA.rand(W, H, 1, N, B)

CUDA.@time plane_volume_rendering(rgb, sigma, xyz_src)



CUDA.allowscalar(false)

xyz_src

permutedims(xyz_src, (3, 1, 2, 4, 5))

poses[1].rvec
poses[1].tvec

rvec = poses[1].rvec
tvec = poses[1].tvec

function get_tgt_xyz_from_plane_disparity(xyz_src, pose)
    # TODO: correct gebruik van pose?
    # pose -> rotation matrix, translation vector: 
    R = so3_exp_map(pose.rvec)
    t = pose.tvec
    
    # R*[x y z]ᵀ + t:
    xyz_src = reshape(permutedims(xyz_src, (3, 1, 2, 4, 5)), (3, :, 4)) # 3×(W×H×N)×B
    xyz_tgt = (R ⊠ xyz_src) .+ t
    return permutedims(reshape(xyz_tgt, (3, W, H, N, B)), (2, 3, 1, 4, 5)) # W×H×3×N×B
end

get_tgt_xyz_from_plane_disparity(xyz_src, poses[1])

@profview get_tgt_xyz_from_plane_disparity(xyz_src, poses[1])

function sample(src, depth_src, pose, K_inv, K)
    # TODO: correct gebruik van pose?
    # pose -> rotation matrix, translation vector: 
    
end

depth_src = 1 ./ mpi_disparity_src

pose = poses[1]

K = train_cache.K
invK = train_cache.invK

R = so3_exp_map(pose.rvec) # rotation tgt -> src of src -> tgt ???
t = pose.tvec

n = transfer([0 0 1]) # TODO: StaticArray?

#TODO: delen door -depth of +depth?

temp = reshape(R, (3, 3, 1, size(R, 3))) .- reshape(t ⊠ n, (3, 3, 1, :)) ./ -reshape(depth_src, (1, 1, size(depth_src)...))

temp

inv(temp[:, :, 1, 1]) # TODO: inverse niet geimplementeerd voor gpu, misschien via expliciete formule voor transformatie?

reshape(K ⊠ reshape(temp, (3, 3, :)) ⊠ invK, (3, 3, size(temp, 3), size(temp, 4)))

function render_tgt(rgb_src, sigma_src, disparity_src, xyz_tgt, pose, K_inv, K)
    depth_src = 1 ./ disparity_src

end
