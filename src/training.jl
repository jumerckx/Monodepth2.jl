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

# function train_loss(
#     model, x::AbstractArray{T}, auto_loss, cache::TrainCache, parameters::Params,
#     do_visualization,
# ) where T
#     target_x = x[:, :, :, cache.target_id, :]
#     disparities, poses = model(x, cache.source_ids, cache.target_id)

#     # TODO pass as parameter to function
#     inverse_transform = cache.source_ids .< cache.target_id
#     Ps = map(
#         p -> composeT(p[1].rvec, p[1].tvec, p[2]),
#         zip(poses, inverse_transform))

#     vis_warped, vis_loss, vis_disparity = nothing, nothing, nothing
#     if do_visualization
#         vis_disparity = cpu(disparities[end])
#     end

#     loss = zero(T)
#     width, height = parameters.target_size

#     for (i, (disparity, scale)) in enumerate(zip(disparities, cache.scales))
#         dw, dh, _, dn = size(disparity)
#         if dw != width || dh != height
#             disparity = upsample_bilinear(disparity; size=(width, height))
#         end

#         depth = disparity_to_depth(
#             disparity, parameters.min_depth, parameters.max_depth)
#         coordinates = cache.backprojections(
#             reshape(depth, (1, width * height, dn)), cache.invK)
#         warped_images = map(zip(Ps, cache.source_ids)) do t
#             uvs = reshape(
#                 cache.projections(coordinates, cache.K, t[1]...),
#                 (2, width, height, dn))
#             grid_sample(x[:, :, :, t[2], :], uvs; padding_mode=:border)
#         end

#         warp_loss = prediction_loss(cache.ssim, warped_images, target_x)
#         if parameters.automasking
#             warp_loss = _apply_mask(auto_loss, warp_loss)
#         end

#         normalized_disparity = (
#             disparity ./ (mean(disparity; dims=(1, 2)) .+ T(1e-7)))[:, :, 1, :]
#         disparity_loss = smooth_loss(normalized_disparity, target_x) .*
#             T(parameters.disparity_smoothness) .* T(scale)

#         loss += mean(warp_loss) + disparity_loss

#         if do_visualization && i == length(cache.scales)
#             vis_warped = cpu.(warped_images)
#             vis_loss = cpu(warp_loss)
#         end
#     end

#     loss / T(length(cache.scales)), vis_disparity, vis_warped, vis_loss
# end

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

function loss_per_scale(src_img, src_depth, tgt_img, scale, K, mpi_rgb, mpi_sigma, disparity, pose; valid_mask_threshold=2)
    W, H, _, N, B = size(mpi_rgb)

    K = ignore_derivatives() do
        K = K ./ eltype(K)(2^scale)
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

    # render_novel_view:
    tgt_imgs_syn, tgt_disparity_syn, tgt_mask_syn = render_novel_view(
        mpi_rgb,
        mpi_sigma,
        disparity,
        pose,
        K_inv,
        K)

    rgb_tgt_valid_mask = tgt_mask_syn .>= tgt_mask_syn
    loss_map = abs.(tgt_imgs_syn .- tgt_img) .* rgb_tgt_valid_mask # TODO: network output wordt niet upscaled zoals in Monodepth?
    loss_rgb_tgt = mean(loss_map) # TODO: met mean wordt de som gedeeld door het totaal aantal pixels, niet het aantal pixels dat binnen de mask zit.
    loss_ssim_tgt = 1 - mean(transfer(SSIM())(tgt_imgs_syn, tgt_img)) # TODO: SSIM meegeven als argument
    loss_smooth_src = smooth_loss(src_disparity_syn[:, :, 1, :], src_img)
    loss_smooth_tgt = smooth_loss(tgt_disparity_syn[:, :, 1, :], tgt_img)
    mask = src_depth .!= 0
    loss_depth = D(src_depth_syn[mask], src_depth[mask])
    # loss_depth = 0
    return src_disparity_syn, loss_rgb_tgt, loss_ssim_tgt, loss_smooth_src, loss_smooth_tgt, loss_depth
end
function train_loss(
    model, src_img::AbstractArray{T}, src_depth::AbstractArray{T}, tgt_img::AbstractArray{T}, pose, K, invK, scales;
    N=32, near=1, far=0.001
) where {T}
    W, H, _, B = size(src_img)
    disparities = Monodepth.uniformly_sample_disparity_from_linspace_bins(N, B; near=eltype(src_img)(near), far=eltype(src_img)(far))
    mpi = model(
        src_img,
        disparities;
        num_bins=N)

    ℒ = 0
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
        ℒ += sum((loss_depth))
    end
    return ℒ, src_disparity_syn
end

d(ŷ, y) = log(ŷ) - log(y)

function D(ŷ, y)
    ds = d.(ŷ, y)
    mean(ds .^ 2) - mean(ds)^2
end