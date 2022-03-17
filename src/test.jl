W, H = 416, 132
N, B = 32, 4

invK = CUDA.rand(3, 3)


function create_meshgrid(H, W)
    permutedims(cat((1:W) .* ones(H)', (1:H)' .* ones(W), ones(W, H), dims=3), (3, 1, 2)) # TODO: offset van 0.5px?
end

create_meshgrid(H, W)

mpi_disparity_src = uniformly_sample_disparity_from_linspace_bins(N, B)

meshgrid_src_homo = transfer(create_meshgrid(H, W))

function get_src_xyz_from_plane_disparity(meshgrid_src_homo, mpi_disparity_src, K_src_inv)
    N, B = size(mpi_disparity_src)

    mpi_depth_src = reshape(1 ./ mpi_disparity_src, (1, 1, 1, N, B))
    return reshape(K_src_inv * reshape(meshgrid_src_homo, 3, :), (3, W, H)) .* mpi_depth_src
end

using LinearAlgebra
import LinearAlgebra:norm
norm(x; dims=:) = sqrt.(sum(abs2.(x), dims=dims))

CUDA.@time xyz_src = get_src_xyz_from_plane_disparity(meshgrid_src_homo, mpi_disparity_src, invK)

function plane_volume_rendering(rgb, sigma, xyz)
    diff = permutedims(xyz[:, :, :, 2:end,:] .- xyz[:, :, :, 1:end-1, :], (2, 3, 1, 4, 5))
    dist = cat(norm(diff, dims=3), CUDA.fill(1f3, W, H, 1, 1, B), dims=4) # TODO: TinyNERF gebruikt fill met 1e10

    transparency = exp.(-dist .* sigma)
    alpha = 1 .- transparency

    transparency_acc = cumprod(transparency .+ 1e-6, dims=4) # TODO: is ".+ 1e-6 " nodig?
    transparency_acc = cat(CUDA.ones(W, H, 1, 1, B), transparency_acc[:, :, :, 1:end-1, :], dims=4)

    weights = transparency_acc .* alpha  # BxSx1xHxW

    rgb_out = dropdims(sum(weights .* rgb, dims=4), dims=4)

    # TODO: return depth?
    return rgb_out, transparency_acc, weights
end

rgb, sigma = CUDA.rand(W, H, 3, N, B), CUDA.rand(W, H, 1, N, B)

size(rgb)
size(sigma)
size(xyz_src)

CUDA.@time plane_volume_rendering(rgb, sigma, xyz_src)

function get_tgt_xyz_from_plane_disparity(xyz_src, pose)
    # TODO: correct gebruik van pose?
    # pose -> rotation matrix, translation vector: 
    R = so3_exp_map(pose.rvec)
    t = pose.tvec
    
    # R*[x y z]ᵀ + t:
    xyz_src = reshape(xyz_src, (3, :, 4)) # 3×(W×H×N)×B
    xyz_tgt = (R ⊠ xyz_src) .+ t
    return reshape(xyz_tgt, (3, W, H, N, B)) # 3×W×H×N×B
end

xyz_src

CUDA.@time get_tgt_xyz_from_plane_disparity(xyz_src, poses[1])

function sample(src, depth_src, pose, K, K_inv)
    R = so3_exp_map(pose.rvec)
    t = pose.tvec
    n = transfer([0 0 1])

    temp = reshape(unsqueeze(t ⊠ n, 3) ./ -reshape(depth_src, (1, 1, size(depth_src, 1), size(depth_src, 2))), (3, 3, :))
    
    H_tgt_src = K ⊠ (R .- temp) ⊠ K_inv
    
    H_src_tgt = inv(H_tgt_src)
    
    meshgrid_tgt_homo = transfer(reshape(create_meshgrid(H, W), (3, :)) .- [1, 1, 0]) # TODO: misschien beter cachen?
    meshgrid_src = H_src_tgt ⊠ meshgrid_tgt_homo
    
    meshgrid_src = meshgrid_src[1:2, :, :] ./ meshgrid_src[3:3, :, :]
    
    valid_mask = (meshgrid_src[1, :, :] .< W .* meshgrid_src[1, :, :] .>= 0) .* (meshgrid_src[2, :, :] .< H .* meshgrid_src[2, :, :] .>= 0)
    
    meshgrid_src[1, :, :] .= (meshgrid_src[1, :, :] .+ eltype(meshgrid_src)(0.5)) ./ (W/2)
    meshgrid_src[2, :, :] .= (meshgrid_src[2, :, :] .+ eltype(meshgrid_src)(0.5)) ./ (H/2)
    
    meshgrid_src = reshape(meshgrid_src, (2, W, H, :))
        
    tgt = grid_sample(src, meshgrid_src; padding_mode=:border)
    return tgt, valid_mask
end

# depth_src = 1 ./ mpi_disparity_src

# pose = poses[1]

# K = train_cache.K
# invK = train_cache.invK

# R = CuArray(I(3)) #so3_exp_map(pose.rvec) # rotation tgt -> src of src -> tgt ???
# t = pose.tvec
# t .= CuArray([0, 0, -0.09])

# n = transfer([0 0 1]) # TODO: StaticArray?

# #TODO: delen door -depth of +depth?

# temp = reshape(R, (3, 3, 1, size(R, 3))) .- reshape(t ⊠ n, (3, 3, 1, :)) ./ -reshape(depth_src, (1, 1, size(depth_src)...))

# temp

# H_tgt_src = reshape(reshape(temp, (3, 3, :)) ⊠ invK, (3, 3, 32, 4))
# # H_tgt_src = inv(H_src_tgt[:, :, 1])

# H_tgt_src = mapslices(inv, collect(H_src_tgt), dims=(1, 2)) # voor effjes!!

# meshgrid = transfer(create_meshgrid(H, W)) # 3 × 

# reshape(meshgrid, (3, :))


# meshgrid_src_homo = reshape(reshape(H_src_tgt, (3, 3, :)) ⊠ reshape(meshgrid, (3, :)), (3, W, H, :))

# meshgrid_src = meshgrid_src_homo[1:2, :, :, :] ./ meshgrid_src_homo[3:3, :, :, :]

# CUDA.@time @views (meshgrid_src[1, :, :, :] .<= W) .* (meshgrid_src[1, :, :, :] .>= 1)
# CUDA.@time @views (meshgrid_src[2, :, :, :] .<= H) .* (meshgrid_src[2, :, :, :] .>= 1)

# K = CuArray(Float32.([192.   0. 192.
# 0. 192. 128.
# 0.   0.   1.]))

# K_inv = inv!(deepcopy(K))

# inv!(K)*K

# inv!(K)

# reshape(H_src_tgt, (3, 3, :))

# eachslice(reshape(H_src_tgt, (3, 3, :)), dims=3)

# CUDA.@time collect(eachslice(reshape(H_src_tgt, (3, 3, :)), dims=3))

# CUDA.@time collect(eachslice(a, dims=3))

# a = CUDA.rand(3, 3, 1000)
# CUDA.@time CUDA.CUBLAS.getrf_batched!(collect(eachslice(a, dims=3)), true)

# a

# a
# CUDA.@time CUDA.CUBLAS.getrf_batched!([a], true)

# K_inv = [0.0052  0.0000 -1.0000
# 0.0000  0.0052 -0.6667
# 0.0000  0.0000  1.0000]

# K*gpu(K_inv)

# CUDA.LAPACK.LAPACK.getri

# meshgrid_src[1, :, :, :]

# (meshgrid_src[1, :, :, :] .<= W)

# H_src_tgt

# H_tgt_src

# true *true

# depth

# collect(eachslice(H_src_tgt, dims=4))

# using CUDA

# CUDA.CUBLAS.LAPACK.getrf!(CUDA.rand(3, 3))
# CUDA.CUBLAS.LAPACK.getri!(CUDA.rand(3, 3))
# using LinearAlgebra

# a = CUDA.rand(3, 3)

# CUDA.CUBLAS.getrf_batched!([a], true)[2]

# function inv!(x)
#     pivot, info = CUDA.CUBLAS.getrf_batched!([x], true)
#     @show size(x)
#     _,_,x = CUDA.CUBLAS.getri_batched([x], pivot)
#     return x[1]
# end

# inv!(a)

# # 3 × HW × B
# inv(temp[:, :, 1, 1]) # TODO: inverse niet geimplementeerd voor gpu, misschien via expliciete formule voor transformatie?

# reshape(K ⊠ reshape(temp, (3, 3, :)) ⊠ invK, (3, 3, size(temp, 3), size(temp, 4)))

# function render_tgt(rgb_src, sigma_src, disparity_src, xyz_tgt, pose, K_inv, K)
#     depth_src = 1 ./ disparity_src

# end

# (t ⊠ n)
# depth_src

