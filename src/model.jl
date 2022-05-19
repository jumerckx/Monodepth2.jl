include("./repeat.jl")
Base.repeat(a::CUDA.GPUArrays.AbstractGPUArray{T, N} where {T, N}, counts::Integer...) = _repeat(a, counts...)

function embed(x; L=10)
    # x should be a matrix of size num_bins×batch_size
    num_bins, batch_size = size(x)
    out = similar(x, (1, 1, 2*L+1, num_bins, batch_size))
    out[1, 1, 1, :, :] .= x

    x = reshape(x, (1, 1, 1, num_bins, batch_size))
    out[:, :, 2:end, :, :] .= cat((f.(2^i .* x) for i in 0:(L - 1) for f in (sin, cos))..., dims=3)

    return out
end
@non_differentiable embed(::Any...)

struct Model{E, D, P}
    encoder::E
    depth_decoder::D
    pose_decoder::P
end
Flux.@functor Model

function (m::Model)(x, disparity; num_bins=32)
    W, H, C, B = size(x)
    # x_flattened = reshape(x, (W, H, C, B))

    features = m.encoder(x, Val(:stages))

    embedded_disparity = embed(disparity)
    depth_decoder_x = map(features) do f
        f = cat(
            repeat(Flux.unsqueeze(f, 4), 1, 1, 1, num_bins, 1),
            repeat(embedded_disparity, size(f, 1), size(f, 2)),
            dims=3)
        reshape(f, size(f, 1), size(f, 2), size(f, 3), size(f, 4)*size(f, 5))
    end



    disparities = m.depth_decoder(depth_decoder_x)
    disparities = map(disparities) do d
        W,H,_,_ = size(d)
        d = reshape(d, (W, H, 4, num_bins, B))
        disparities_rgb = sigmoid.(d[:,:,1:3,:, :])
        disparities_sigma = relu.(d[:, :, 4:4, :, :]) .+ eltype(d)(1e-4) # zie depth_decoder.py:139, +1e-4 is nodig om 0'en in sigma te vermijden (anders NaN-loss)
        (disparities_rgb, disparities_sigma)
    end
    
    # poses = eval_poses(m, features[end], source_ids, target_id)
    disparities #, poses
end

function eval_poses(m::Model, features, source_ids, target_id)
    map(
        i -> m.pose_decoder(_get_pose_features(features, i, target_id)),
        source_ids)
end

eval_disparity(m::Model, x) = m.depth_decoder(m.encoder(x, Val(:stages)))

function _get_pose_features(features, i, target_id)
    if i < target_id
        return features[:, :, :, i, :], features[:, :, :, target_id, :]
    end
    features[:, :, :, target_id, :], features[:, :, :, i, :]
end
