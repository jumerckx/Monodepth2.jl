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

function uniformly_sample_disparity_from_linspace_bins(num_bins, batch_size; near=1f0, far=0.001f0)
    bin_edges_start = range(near, far, num_bins+1)[1:end-1]
    interval = bin_edges_start[2]-bin_edges_start[1]
    return bin_edges_start .+ (CUDA.rand(num_bins, batch_size)*interval)
end
@non_differentiable uniformly_sample_disparity_from_linspace_bins(::Any...)

struct Model{E, D, P}
    encoder::E
    depth_decoder::D
    pose_decoder::P
end
Flux.@functor Model

function (m::Model)(x, source_ids, target_id; num_bins=32)
    w, h, c, l, n = size(x)
    x_flattened = reshape(x, (w, h, c, l * n))

    features = map(
        f -> reshape(f, (size(f, 1), size(f, 2), size(f, 3), l, n)),
        m.encoder(x_flattened, Val(:stages)))

    embedded_disparity = embed(uniformly_sample_disparity_from_linspace_bins(num_bins, n))

    depth_decoder_x = map(features) do f
        f = cat(
            repeat(f[:, :, :, target_id:target_id, :], 1, 1, 1, num_bins, 1),
            repeat(embedded_disparity, size(f, 1), size(f, 2)),
            dims=3
        ) # H×W×(channels+embedding)×planes×batch_size

        # merge plane- and batch-dimension together:
        reshape(f, size(f, 1), size(f, 2), size(f, 3), size(f, 4)*size(f, 5))
    end
    
    disparities = m.depth_decoder(depth_decoder_x)
    poses = eval_poses(m, features[end], source_ids, target_id)
    disparities, poses
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
