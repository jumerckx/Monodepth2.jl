import LinearAlgebra

function inv!(x)
    pivot, info = CUDA.CUBLAS.getrf_batched!(x, true)
    _,_,x = CUDA.CUBLAS.getri_batched(x, pivot)
    return x
end
function inv!(x::CuArray{T, 2}) where T
    return inv!([x])[1]
end
function inv!(x::CuArray{T, 3}) where T
    cat(inv!(collect(eachslice(x, dims=3)))..., dims=3) # TODO: moet sneller kunnen
end
function inv(x::CuArray{T, N}) where {T, N}
    x = deepcopy(x)
    return inv!(x)
end

norm(x; dims) = sqrt.(sum(abs2.(x), dims=dims))

function create_meshgrid(H, W)
    permutedims(cat((1:W) .* ones(H)', (1:H)' .* ones(W), ones(W, H), dims=3), (3, 1, 2)) # TODO: offset van 0.5px?
end

function get_src_xyz_from_plane_disparity(meshgrid_src_homo, mpi_disparity_src, K_src_inv)
    N, B = size(mpi_disparity_src)
    _, W, H = size(meshgrid_src_homo)
    mpi_depth_src = reshape(1 ./ mpi_disparity_src, (1, 1, 1, N, B))
    return reshape(K_src_inv * reshape(meshgrid_src_homo, 3, :), (3, W, H)) .* mpi_depth_src
end

function plane_volume_rendering(rgb, sigma, xyz)
    W, H, _, N, B = size(rgb)
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

function get_tgt_xyz_from_plane_disparity(xyz_src, pose)
    # TODO: correct gebruik van pose?
    # pose -> rotation matrix, translation vector: 
    R = so3_exp_map(pose.rvec)
    t = pose.tvec
    
    _ , W, H, N, B = size(xyz_src)
    # R*[x y z]ᵀ + t:
    xyz_src = reshape(xyz_src, (3, :, B)) # 3×(W×H×N)×B
    xyz_tgt = (R ⊠ xyz_src) .+ reshape(t, (3, 1, :))
    return  reshape(xyz_tgt, (3, W, H, N, B)) # 3×W×H×N×B
end

function sample(src, depth_src, pose, K, K_inv)
    W, H, _, _ = size(src)
    R = so3_exp_map(pose.rvec)
    t = Flux.unsqueeze(pose.tvec, 2)
    n = transfer([0 0 1])

    temp = Flux.unsqueeze(t ⊠ n, 3) ./ -reshape(depth_src, (1, 1, size(depth_src, 1), size(depth_src, 2)))
    # @show size(te mp), size(R)
    H_tgt_src = K ⊠ (reshape(Flux.unsqueeze(R, 3) .- temp, (3, 3, :))) ⊠ K_inv
    
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

function render_tgt_rgb_depth(rgb, sigma, disparity_src, xyz_tgt, pose, K_inv, K)
    # size(rgb) = (W, H, 3, N, B)
    # size(sigma) = (W, H, 1, N, B)
    # size(disparity_src) = (N, B)
    # size(xyz_tgt) = (W, H, 3, N, B) ??
    # size(K_inv) = (3, 3)
    # size(K) = (3, 3)

    W, H, _, N, B = size(rgb)

    depth_src = 1 ./ disparity_src
    xyz_src = cat(rgb, sigma, xyz_tgt, dims=3)

    tgt, valid_mask = sample(reshape(xyz_src, (W, H, 7, N*B)), depth_src, pose, K, K_inv)

    xyz = tgt[:, :, ]


end
