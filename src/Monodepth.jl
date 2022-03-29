module Monodepth

export Depth10k, KittyDataset, SupervisedKITTI, NYUDataset, DChain, DataLoaders # dataloading
export Params, TrainCache
export save_disparity # io_utils.jl
export SSIM # utils.jl
export DepthDecoder, PoseDecoder, ResNet, Model
export train_loss # training.jl

using LinearAlgebra
using Printf
using Statistics

using Augmentations
# import BSON
# using BSON: @save, @load
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

import Random
# Random.seed!(42)

Base.@kwdef struct Params
    min_depth::Float64 = 0.1
    max_depth::Float64 = 100.0
    disparity_smoothness::Float64 = 1e-3
    frame_ids::Vector{Int64} = [1, 2, 3]

    automasking::Bool = true

    target_size::Tuple{Int64, Int64} # (width, height)
    batch_size::Int64
end

struct TrainCache{S}
    ssim::S
    scales::Vector{Float64}
end

include("dtk.jl")
include("kitty.jl")
include("nyudataset.jl")
include("supervisedKITTI.jl")
include("dchain.jl")

include("io_utils.jl")
include("utils.jl")
include("render.jl")
include("depth_decoder.jl")
include("pose_decoder.jl")
include("model.jl")
include("simple_depth.jl")

include("training.jl")

end
