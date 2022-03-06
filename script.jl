using LinearAlgebra
using Printf
using Statistics

using Augmentations
import BSON
using BSON: @save, @load
using DataLoaders
using FileIO
using ImageCore
using ImageTransformations
using MLDataPattern: shuffleobs
using VideoIO
using ProgressMeter
using Plots
using StaticArrays
gr()

import ChainRulesCore: rrule
using ChainRulesCore
using CUDA
using Flux
using ResNet

Base.@kwdef struct Params
    min_depth::Float64 = 0.1
    max_depth::Float64 = 100.0
    disparity_smoothness::Float64 = 1e-3
    frame_ids::Vector{Int64} = [1, 2, 3]

    automasking::Bool = true

    target_size::Tuple{Int64, Int64} # (width, height)
    batch_size::Int64
end

struct TrainCache{S, B, P, I}
    ssim::S
    backprojections::B
    projections::P

    K::I
    invK::I

    target_id::Int64
    source_ids::Vector{Int64}
    scales::Vector{Float64}
end

include("src/dtk.jl")
include("src/kitty.jl")
include("src/dchain.jl")

include("src/io_utils.jl")
include("src/utils.jl")
include("src/depth_decoder.jl")
include("src/pose_decoder.jl")
include("src/model.jl")
include("src/simple_depth.jl")

include("src/training.jl")

device = gpu
precision = f32
transfer = device ∘ precision
@show transfer

log_dir = "./out/logs"
save_dir = "./out/models"

isdir(log_dir) || mkpath(log_dir)
isdir(save_dir) || mkpath(save_dir)

grayscale = true
in_channels = grayscale ? 1 : 3
augmentations = FlipX(0.5)
target_size=(128, 416)

# kitty_dir = "/home/pxl-th/projects/datasets/kitty-dataset"
# datasets = [
#     KittyDataset(kitty_dir, s; target_size, augmentations)
#     for s in map(i -> @sprintf("%02d", i), 0:21)]

datasets = []
dtk_dir = "../depth10k"
dtk_dataset = Depth10k(
    joinpath(dtk_dir, "imgs"),
    readlines(joinpath(dtk_dir, "trainable-nonstatic"));
    augmentations, grayscale)
push!(datasets, dtk_dataset)

dchain = DChain(datasets)
dataset = datasets[begin]

width, height = dataset.resolution
parameters = Params(;
    batch_size=4, target_size=dataset.resolution,
    disparity_smoothness=1e-3, automasking=false)
max_scale, scale_levels = 5, collect(2:5)
scales = [1.0 / 2.0^(max_scale - level) for level in scale_levels]
println(parameters)

train_cache = TrainCache(
    transfer(SSIM()),
    transfer(Backproject(; width, height)),
    transfer(Project(; width, height)),
    transfer(Array(dataset.K)), transfer(Array(dataset.invK)),
    dataset.target_id, dataset.source_ids, scales)

train_cache.backprojections.coordinates
train_cache.projections.normalizer
train_cache.K
train_cache.K

encoder = ResidualNetwork(18; in_channels, classes=nothing)
encoder_channels = collect(encoder.stages)
model = transfer(Model(
    encoder,
    DepthDecoder(;encoder_channels, scale_levels, embedding_levels=21),
    PoseDecoder(encoder_channels[end])))

θ = Flux.params(model)
optimizer = ADAM(1e-4)
trainmode!(model)

# Test:

x = transfer(first(DataLoader(dchain, 4)))

CUDA.allowscalar(false)
disparities, poses = CUDA.@time model(x, train_cache.source_ids, train_cache.target_id)

disparities

disparities = map(disparities) do d
    W, H, C, _ = size(d)
    reshape(d, (W, H, C, 32, :))
end

size.(disparities)
