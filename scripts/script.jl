using LinearAlgebra
using Printf
using Statistics

using Monodepth, Augmentations
# import Monodepth:BSON
using BSON: @save, @load
using DataLoaders
# using FileIO
# using ImageCore
# using ImageTransformations
# using MLDataPattern: shuffleobs
# using VideoIO
# using ProgressMeter
# using Plots
# using StaticArrays
# gr()

# import ChainRulesCore: rrule
# using ChainRulesCore
using CUDA
using Flux
using Monodepth.ResNet

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
    transfer(Monodepth.Backproject(; width, height)),
    transfer(Monodepth.Project(; width, height)),
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

Monodepth.f(10)

disparities

disparities = map(disparities) do d
    W, H, C, _ = size(d)
    reshape(d, (W, H, C, 32, :))
end

Monodepth.

Monodepth.get_src_xyz_from_plane_disparity
