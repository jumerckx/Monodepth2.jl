import LinearAlgebra: inv, inv!

function LinearAlgebra.inv!(x::Vector{CuArray{T, N, M}}) where {T, N, M}
    pivot, info = CUDA.CUBLAS.getrf_batched!(x, true)
    _,_,x = CUDA.CUBLAS.getri_batched(x, pivot)
    return x
end
function LinearAlgebra.inv!(x::CuArray{T, 2}) where T
    return inv!([x])[1]
end
function LinearAlgebra.inv!(x::CuArray{T, 3}) where T
    cat(inv!(collect(eachslice(x, dims=3)))..., dims=3) # TODO: moet sneller kunnen
end
function LinearAlgebra.inv(x::CuArray{T, N}) where {T, N}
    x = deepcopy(x)
    return inv!(x)
end
function ChainRulesCore.rrule(::typeof(LinearAlgebra.inv), x::CuArray{T, 3}) where T
    Ω = inv(x)
    function inv_pullback(ΔΩ)
        return NoTangent(), -permutedims(Ω, (2, 1, 3)) ⊠ ΔΩ ⊠ permutedims(Ω, (2, 1, 3))
    end
    return Ω, inv_pullback
end

function uniformly_sample_disparity_from_linspace_bins(num_bins, batch_size; near=1f0, far=0.001f0)
    bin_edges_start = range(near, far, num_bins+1)[1:end-1]
    interval = bin_edges_start[2]-bin_edges_start[1]
    return bin_edges_start .+ (CUDA.rand(num_bins, batch_size)*interval)
end
@non_differentiable uniformly_sample_disparity_from_linspace_bins(::Any...)


norm(x; dims) = sqrt.(sum(abs2.(x), dims=dims))

function create_meshgrid(H, W)
    permutedims(cat((1:W) .* ones(H)', (1:H)' .* ones(W), ones(W, H), dims=3), (3, 1, 2)) # TODO: offset van 0.5px?
end

function get_src_xyz_from_plane_disparity(meshgrid_src_homo, mpi_disparity_src, K_src_inv)
    N, B = size(mpi_disparity_src)
    _, W, H = size(meshgrid_src_homo)
    mpi_depth_src = reshape(1 ./ mpi_disparity_src, (1, 1, 1, N, B))
    xyz_src = reshape(K_src_inv * reshape(meshgrid_src_homo, 3, :), (3, W, H)) .* mpi_depth_src
    return xyz_src
end

function plane_volume_rendering(rgb, sigma, xyz)
    W, H, _, N, B = size(rgb)
    dist = ignore_derivatives() do
        diff = permutedims(xyz[:, :, :, 2:end,:] .- xyz[:, :, :, 1:end-1, :], (2, 3, 1, 4, 5))
        dist = cat(norm(diff, dims=3), CUDA.fill(1f3, W, H, 1, 1, B), dims=4) # TODO: TinyNERF gebruikt fill met 1e10
        return dist
    end
    transparency = exp.(-dist .* sigma)
    alpha = 1 .- transparency

    transparency_acc = cumprod(transparency .+ eltype(transparency)(1e-6), dims=4) # TODO: is ".+ 1e-6 " nodig?
    cu_ones = ignore_derivatives() do
        CUDA.ones(W, H, 1, 1, B)
    end
    transparency_acc = cat(cu_ones, transparency_acc[:, :, :, 1:end-1, :], dims=4)

    weights = transparency_acc .* alpha
    
    rgb_out, depth_out = weighted_sum_mpi(rgb, xyz, weights)

    return rgb_out, depth_out, transparency_acc, weights
end

function weighted_sum_mpi(rgb, xyz, weights)
    rgb_out = dropdims(sum(weights .* rgb, dims=4), dims=4)
    
    # assume is_bg_depth_inf == false:
    weights_sum = dropdims(sum(weights, dims=4), dims=4) # sum over planes
    depth_out = dropdims(sum(weights .* permutedims(xyz[3:3, :, :, :, :], (2, 3, 1, 4, 5)), dims=4), dims=4) ./ (weights_sum .+ 1e-5)
    
    return rgb_out, depth_out
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
    # @show size(temp), size(R)
    H_tgt_src = K ⊠ (reshape(Flux.unsqueeze(R, 3) .- temp, (3, 3, :))) ⊠ K_inv
    
    H_src_tgt = inv(H_tgt_src)
    meshgrid_tgt_homo = ignore_derivatives() do
        transfer(reshape(create_meshgrid(H, W), (3, :)) .- [1, 1, 0]) # TODO: misschien beter cachen?
    end
    meshgrid_src = H_src_tgt ⊠ meshgrid_tgt_homo
    
    meshgrid_src = meshgrid_src[1:2, :, :] ./ meshgrid_src[3:3, :, :]
    
    valid_mask = (meshgrid_src[1, :, :] .< W .* meshgrid_src[1, :, :] .>= 0) .* (meshgrid_src[2, :, :] .< H .* meshgrid_src[2, :, :] .>= 0)
    
    # meshgrid_src[1, :, :] .= (meshgrid_src[1, :, :] .+ eltype(meshgrid_src)(0.5)) ./ (W/2)
    # meshgrid_src[2, :, :] .= (meshgrid_src[2, :, :] .+ eltype(meshgrid_src)(0.5)) ./ (H/2)

    meshgrid_src = vcat((meshgrid_src[1:1, :, :] .+ eltype(meshgrid_src)(0.5)) ./ (W/2), (meshgrid_src[2:2, :, :] .+ eltype(meshgrid_src)(0.5)) ./ (H/2))

    meshgrid_src = reshape(meshgrid_src, (2, W, H, :))
        
    tgt = grid_sample(src, meshgrid_src; padding_mode=:border)
    return tgt, valid_mask
end

function render_tgt_rgb_depth(rgb, sigma, disparity_src, xyz_tgt, pose, K_inv, K)
    # size(rgb) = (W, H, 3, N, B)
    # size(sigma) = (W, H, 1, N, B)
    # size(disparity_src) = (N, B)
    # size(xyz_tgt) = (3, W, H, N, B) ??
    # size(K_inv) = (3, 3)
    # size(K) = (3, 3)
    W, H, _, N, B = size(rgb)

    depth_src = 1 ./ disparity_src
    xyz_src = cat(rgb, sigma, permutedims(xyz_tgt, (2, 3, 1, 4, 5)), dims=3)
    tgt, valid_mask = sample(reshape(xyz_src, (W, H, 7, N*B)), depth_src, pose, K, K_inv)
    tgt = reshape(tgt, (W, H, :, N, B))
    valid_mask = reshape(valid_mask, (W, H, :, N, B))
    rgb = tgt[:, :, 1:3, :, :]
    sigma = tgt[:, :, 4:4, :, :]
    xyz = tgt[:, :, 5:end, :, :]
    sigma = sigma .* (sigma .>= 0)
    rgb, depth, _ = plane_volume_rendering(rgb, sigma, permutedims(xyz, (3, 1, 2, 4, 5)))
    mask = sum(valid_mask, dims=4)

    return rgb, depth, mask
end

function render_novel_view(mpi_rgb, mpi_sigma, disparity_src, pose, K_inv, K; scale=0)
    W, H, _, N, B = size(mpi_rgb)
    meshgrid = create_meshgrid(H, W)|>transfer # TODO: misschien beter als argument?
    xyz_src = get_src_xyz_from_plane_disparity(meshgrid, disparity_src, K_inv)
    xyz_tgt = get_tgt_xyz_from_plane_disparity(xyz_src, pose)
    tgt_imgs_syn, tgt_depth_syn, tgt_mask_syn = render_tgt_rgb_depth(mpi_rgb, mpi_sigma, disparity_src, xyz_tgt, pose, K_inv, K)
    tgt_disparity_syn = 1 ./ tgt_depth_syn

    return tgt_imgs_syn, tgt_disparity_syn, tgt_mask_syn
end
