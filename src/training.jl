function photometric_loss(
    ssim, predicted::AbstractArray{T}, target::AbstractArray{T}; α=T(0.85)
) where {T}
    l1_loss = mean(abs.(target .- predicted); dims=3)
    ssim_loss = mean(ssim(predicted, target); dims=3)
    α .* ssim_loss .+ (one(T) - α) .* l1_loss
end

@inline function automasking_loss(ssim, inputs, target::T; source_ids)::T where {T}
    minimum(cat(map(i -> photometric_loss(ssim, inputs[:, :, :, i, :], target), source_ids)...; dims=3); dims=3)
end

@inline function prediction_loss(ssim, predictions, target::T)::T where {T}
    minimum(cat(map(p -> photometric_loss(ssim, p, target), predictions)...; dims=3); dims=3)
end

@inline function _apply_mask(mask::T, warp_loss::T)::T where {T}
    minimum(cat(mask, warp_loss; dims=3); dims=3)
end

function network_forward(model, rgb; N=32, K_inv)
    W, H, _, B = size(rgb)

    meshgrid_src = gpu(create_meshgrid(H, W))
    disparity_src = uniformly_sample_disparity_from_linspace_bins(N, B; near=1.0f0, far=0.001f0)

    # _get_disparity_list
    xyz_src = get_src_xyz_from_plane_disparity(
        meshgrid_src,
        disparity_src,
        K_inv
    )

    #disparities, poses = model(rgb, disparity_src, nothing, nothing)
    mpi = model(rgb, disparity_src)
    return mpi
end

using Infiltrator
function loss_per_scale(src_img, src_depth, tgt_img, scale, K, mpi_rgb, mpi_sigma, disparity, pose; valid_mask_threshold=2)
    W, H, _, N, B = size(mpi_rgb)

    K = ignore_derivatives() do
        K = K .* eltype(K)(scale)
        CUDA.@allowscalar K[3, 3] = 1
        return K
    end

    # TODO: K_inv kan misschien efficienter bepaald worden?
    K_inv = inv(K)

    meshgrid = create_meshgrid(H, W) |> transfer
    xyz_src = get_src_xyz_from_plane_disparity(meshgrid, disparity, K_inv)

    src_img_syn, src_depth_syn, blend_weights, weights = plane_volume_rendering(mpi_rgb, mpi_sigma, xyz_src)
    mpi_rgb = blend_weights .* Flux.unsqueeze(src_img, 4) .+ (1 .- blend_weights) .* mpi_rgb
    src_img_syn, src_depth_syn = weighted_sum_mpi(mpi_rgb, xyz_src, weights)
    src_disparity_syn = 1 ./ src_depth_syn

    # TODO: scale_factor?
    # mask = .!iszero.(src_depth)
    # src_pt3d_disp = CUDA.zeros(3, W, H, B)
    # src_pt3d_disp[1, :, :, :] .= 1:W
    # src_pt3d_disp[2, :, :, :] .= (1:H)'
    # src_pt3d_disp[3, :, :, :] .= src_depth[:, :, 1, :]
    # src_pt3d_disp = reshape(src_pt3d_disp, (3, :))
    # src_pt3d_disp = reshape(K * src_pt3d_disp, (3, W, H, B))
    # src_pt3d_disp = src_pt3d_disp[1:2, :, :, :] ./ src_pt3d_disp[3:3, :, :, :]
    
    # poging:
    mask, ds = ignore_derivatives() do
        mask = src_depth .!= 0
        ds = d.(src_depth_syn[mask], src_depth[mask])
        s = mean(ds)
        pose.tvec ./= eltype(pose.tvec)(s)
        mask, ds
    end

    # render_novel_view:
    tgt_imgs_syn, tgt_disparity_syn, tgt_mask_syn = render_novel_view(
        mpi_rgb,
        mpi_sigma,
        disparity,
        pose,
        K_inv,
        K)

    rgb_tgt_valid_mask = tgt_mask_syn[:, :, :, 1, :] .>= valid_mask_threshold
    loss_map = abs.(tgt_imgs_syn .- tgt_img) .* rgb_tgt_valid_mask # TODO: network output wordt niet upscaled zoals in Monodepth?
    loss_rgb_tgt = mean(loss_map) # TODO: met mean wordt de som gedeeld door het totaal aantal pixels, niet het aantal pixels dat binnen de mask zit.
    loss_ssim_tgt = 1 - mean(transfer(SSIM())(tgt_imgs_syn .* rgb_tgt_valid_mask, tgt_img .* rgb_tgt_valid_mask)) # TODO: SSIM meegeven als argument
    loss_smooth_src = smooth_loss(src_disparity_syn[:, :, 1, :], src_img)
    loss_smooth_tgt = smooth_loss(tgt_disparity_syn[:, :, 1, :], tgt_img)
    mask = src_depth .!= 0
    loss_depth = D(src_depth_syn[mask], src_depth[mask])
    # loss_depth = 0
    ignore_derivatives() do
        @infiltrate
        @infiltrate loss_smooth_tgt > 100
        @infiltrate isnan(sum((loss_rgb_tgt, loss_ssim_tgt, loss_smooth_src, loss_smooth_tgt, loss_depth))) || any(isnan.(src_disparity_syn))
    end
    return src_disparity_syn, loss_rgb_tgt, loss_ssim_tgt, loss_smooth_src, loss_smooth_tgt, loss_depth
end
function train_loss(
    model, src_img::AbstractArray{T}, src_depth::AbstractArray{T}, tgt_img::AbstractArray{T}, pose, K, invK, scales;
    N=32, near=1, far=0.001
) where {T}

    W, H, _, B = size(src_img)
    minimal_depth = max(near, -minimum(pose.tvec[3, :]) + 0.1) # eerste plane mag vanaf 10cm van de camera liggen.
    disparities = Monodepth.uniformly_sample_disparity_from_linspace_bins(N, B; near=eltype(src_img)(1 / minimal_depth), far=eltype(src_img)(far))
    # disparities = Monodepth.uniformly_sample_disparity_from_linspace_bins(N, B; near=eltype(src_img)(near), far=eltype(src_img)(far))
    mpi = model(
        src_img,
        disparities;
        num_bins=N)

    ℒ = 0
    total_loss_rgb_tgt, total_loss_ssim_tgt, total_loss_smooth_src, total_loss_depth, total_loss_smooth_tgt = 0, 0, 0, 0, 0
    src_disparity_syn = nothing
    for (scale, (mpi_rgb, mpi_sigma)) in zip(scales, mpi)
        src_img_scaled = upsample_bilinear(src_img, scale)
        tgt_img_scaled = upsample_bilinear(tgt_img, scale) # TODO: misschien sneller om src_- en tgt_image tegelijk te downsamplen?
        src_depth_scaled = upsample_bilinear(src_depth, scale)
        src_disparity_syn, loss_rgb_tgt, loss_ssim_tgt, loss_smooth_src, loss_smooth_tgt, loss_depth = loss_per_scale(
            src_img_scaled,
            src_depth_scaled,
            tgt_img_scaled,
            scale,
            K,
            mpi_rgb,
            mpi_sigma,
            disparities,
            pose)
        ℒ += (
            1e-2 * loss_rgb_tgt, # rgb_tgt_syn <-> rgb_tgt
            1e-1 * loss_ssim_tgt, # rgb_tgt_syn <-> rgb_tgt
            # 1e-4 * loss_smooth_src, # disparity_src_syn <-> rgb_src
            # 5e-4 * loss_smooth_tgt, # disparity_tgt_syn <-> rgb_tgt
            loss_depth, # disparity_src_syn <-> disparity_src
        ) |> sum
        # ℒ += loss_depth + loss_rgb_tgt + 1e-2 * loss_smooth_src + loss_ssim_tgt + loss_rgb_tgt
        ignore_derivatives() do
            @infiltrate loss_depth > 10
            @infiltrate isnan(sum((loss_rgb_tgt, loss_ssim_tgt, loss_smooth_src, loss_smooth_tgt, loss_depth)))
        end
        total_loss_depth += loss_depth
        total_loss_rgb_tgt += loss_rgb_tgt
        total_loss_ssim_tgt += loss_ssim_tgt
        total_loss_smooth_tgt += loss_smooth_tgt
        total_loss_smooth_src += loss_smooth_src
    end
    return ℒ, src_disparity_syn, (total_loss_rgb_tgt, total_loss_ssim_tgt, total_loss_smooth_src, total_loss_depth, total_loss_smooth_tgt)
end

d(ŷ, y) = log(ŷ) - log(y)

function D(ŷ, y)
    ds = d.(ŷ, y)
    D(ds)
    # mean(ds .^ 2) - mean(ds)^2
    # mean(ds .^ 2) # gewone MSE
end

function D(ds)
    mean(ds .^ 2) - mean(ds)^2
end