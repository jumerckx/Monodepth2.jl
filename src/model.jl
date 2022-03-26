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

function (m::Model)(x)
    features = m.encoder(x, Val(:stages))
    disparities = m.depth_decoder(features)
    disparities
end

eval_disparity(m::Model, x) = m.depth_decoder(m.encoder(x, Val(:stages)))
