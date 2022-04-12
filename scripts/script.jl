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

grayscale = false
in_channels = grayscale ? 1 : 3
augmentations = FlipX(0.5)
target_size=(128, 416)

kitty_dir = "/scratch/vop_BC04/KITTI/"
# datasets = [
    
#     KittyDataset(kitty_dir, s; target_size, augmentations)
#     for s in map(i -> @sprintf("%02d", i), 0:21)]
datasets = []
for datum in filter(isdir, joinpath.(kitty_dir,  readdir(kitty_dir)))
    datum_path = joinpath(kitty_dir, datum)
    calib_path = joinpath(datum_path, "calib_cam_to_cam.txt")
    for drive in filter(isdir, joinpath.(datum_path,  readdir(datum_path)))
        poses_path = joinpath(drive, "poses.txt")
        push!(datasets, KittyDataset(drive, calib_path, poses_path; target_size, augmentations)) 
    end
end

datasets[1][1]

# datasets = []
# dtk_dir = "../depth10k"
# dtk_dataset = Depth10k(
#     joinpath(dtk_dir, "imgs"),
#     readlines(joinpath(dtk_dir, "trainable-nonstatic"));
#     augmentations, grayscale)
# push!(datasets, dtk_dataset)

dchain = DChain(datasets)
dataset = datasets[begin]

dchain[1]

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

# x = transfer(first(DataLoader(dchain, 3)))

x = CUDA.rand(416, 128, 3, 2)

CUDA.allowscalar(false)
disparities = CUDA.@time model(
    x,
    Monodepth.uniformly_sample_disparity_from_linspace_bins(32, 2; near=1f0, far=0.001f0),
    train_cache.source_ids,
    train_cache.target_id)



mpi = Monodepth.network_forward(
    model,
    x,
    K_inv=CUDA.rand(3, 3))

src_img = x
tgt_img = CUDA.rand(size(src_img)...)




pose = Monodepth.Pose(CUDA.rand(3, 2), CUDA.rand(3, 2))
K = KittiDataSet.K





invK = Monodepth.inv(K)

Monodepth.train_loss(
    model,
    x,
    tgt_img,
    pose,
    K,
    invK,
    scales)

transfer(SSIM())(tgt_img, tgt_img)
