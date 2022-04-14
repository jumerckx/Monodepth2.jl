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


function train(;η=1e-4, model=nothing, θ=nothing)
    device = gpu
    precision = f32
    transfer = device ∘ precision
    @show transfer

    log_dir = "/scratch/vop_BC04/out/logs"
    save_dir = "/scratch/vop_BC04/out/models"

    isdir(log_dir) || mkpath(log_dir)
    isdir(save_dir) || mkpath(save_dir)

    grayscale = false
    in_channels = grayscale ? 1 : 3
    augmentations = FlipX(0.5)
    
    
    target_size= (128,416)
  

    img_dir = "/scratch/vop_BC04/KITTI/"
    depth_dir = "/scratch/vop_BC04/depth_maps/"
    
    datasets = []
    for datum in filter(isdir, joinpath.(img_dir,  readdir(img_dir)))
        datum_path = joinpath(img_dir, datum)
        calib_path = joinpath(datum_path, "calib_cam_to_cam.txt")
        for drive in readdir(datum_path)[isdir.(joinpath.(datum_path,  readdir(datum_path)))] # filter(isdir, joinpath.(datum_path,  readdir(datum_path)))
            drive_depth_path = joinpath(depth_dir, drive, "proj_depth", "groundtruth")
            if (isdir(drive_depth_path))
                drive = joinpath(datum_path, drive)
                poses_path = joinpath(drive, "poses.txt")
                push!(datasets, KittyDataset(drive, drive_depth_path, calib_path, poses_path; target_size, augmentations)) 
            end
        end
    end

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
        transfer(Monodepth.SSIM()),
        transfer(Array(dataset.K)),
        transfer(Array(dataset.invK)),
        scales)
    
    if (isnothing(model))
        encoder = ResidualNetwork(18; in_channels, classes=nothing)
        encoder_channels = collect(encoder.stages)
        model = transfer(Model(
            encoder,
            DepthDecoder(;encoder_channels, scale_levels, embedding_levels=21),
            PoseDecoder(encoder_channels[end])))
    end
    if (isnothing(θ)); θ = Flux.params(model); end

    optimizer = ADAM(η)
    trainmode!(model)

    # Perform first gradient computation using small batch size.
    println("Precompile grads...")
    for x_host in DataLoader(dchain, 1)
        src_img, src_depth, tgt_img, pose = transfer.(x_host)

        println("Forward:")
        @time train_loss(model, src_img, src_depth, tgt_img,
        pose, train_cache.K, train_cache.invK, scales)
        

        println("Backward:")
        @time begin
            ∇ = gradient(θ) do
                losses = train_loss(model, src_img, src_depth, tgt_img,
                pose, train_cache.K, train_cache.invK, scales)
            end
        end

        # @show mean(∇[model.pose_decoder.pose[end].weight])
        break
    end
    GC.gc()

    # Do regular training.
    n_epochs, log_iter, save_iter = 20, 50, 500

    println("Training...")
    for epoch in 1:n_epochs
        loader = DataLoader(shuffleobs(dchain), parameters.batch_size)
        bar = get_pb(length(loader), "Epoch $epoch / $n_epochs: ")

        for (i, x_host) in enumerate(loader)
            src_img, src_depth, tgt_img, pose = transfer.(x_host)

            auto_loss = nothing
            # if parameters.automasking
            #     auto_loss = automasking_loss(
            #         train_cache.ssim, x, x[:, :, :, train_cache.target_id, :];
            #         source_ids=train_cache.source_ids)
            # end

            loss_cpu = 0.0
            disparity, warped, vis_loss = nothing, nothing, nothing
            do_visualization = i % log_iter == 0 || i == 1

            Flux.Optimise.update!(optimizer, θ, gradient(θ) do
                loss, disparity = train_loss(
                    model, src_img, src_depth, tgt_img,
                    pose, dataset.K, dataset.invK, scales)
                loss_cpu = cpu(loss)
                loss
            end)

            if do_visualization
                save_disparity(disparity[:, :, 1, 1])
                colorview(RGB, permutedims(collect(x[:, :, :, 1]), (3, 2, 1)))|>display

                # save_disparity(
                #     disparity[:, :, 1, 1],
                #     joinpath(log_dir, "loss-$epoch-$i.png"))
            end
            if i % save_iter == 0
                model_host = cpu(model)
                # @save joinpath(save_dir, "$epoch-$i-$loss_cpu.bson") model_host
            end

            next!(bar; showvalues=[(:i, i), (:loss, loss_cpu)])
        end
    end
end
CUDA.allowscalar(false)
train()

##############################################################################

# log_dir = "./out/logs"
# save_dir = "./out/models"

# isdir(log_dir) || mkpath(log_dir)
# isdir(save_dir) || mkpath(save_dir)

# grayscale = false
# in_channels = grayscale ? 1 : 3
# augmentations = FlipX(0.5)
# target_size=(128, 416)

# kitty_dir = "/scratch/vop_BC04/KITTI/"
# depth_dir = "/scratch/vop_BC04/depth_maps/"
# # datasets = [
    
# #     KittyDataset(kitty_dir, s; target_size, augmentations)
# #     for s in map(i -> @sprintf("%02d", i), 0:21)]
# datasets = []
# for datum in filter(isdir, joinpath.(kitty_dir,  readdir(kitty_dir)))
#     datum_path = joinpath(kitty_dir, datum)
#     calib_path = joinpath(datum_path, "calib_cam_to_cam.txt")
#     for drive in readdir(datum_path)[isdir.(joinpath.(datum_path,  readdir(datum_path)))] # filter(isdir, joinpath.(datum_path,  readdir(datum_path)))
#         # println(datum_path)
#         # println(drive)
#         drive_depth_path = joinpath(depth_dir, drive, "proj_depth", "groundtruth")
#         if (isdir(drive_depth_path))
#             # println(drive_depth_path)
#             drive = joinpath(datum_path, drive)
#             poses_path = joinpath(drive, "poses.txt")
#             push!(datasets, KittyDataset(drive, drive_depth_path, calib_path, poses_path; target_size, augmentations)) 
#         end
#     end
# end

# [println(d.depth_dir) for d in datasets]

# KittyDataset(joinpath(kitty_dir, "2011_09_26/2011_09_26_drive_0001_sync"), joinpath(depth_dir, "2011_09_26_drive_0001_sync/proj_depth/groundtruth"),
#     joinpath(kitty_dir, "2011_09_26", "calib_cam_to_cam.txt"),
#     joinpath(kitty_dir, "2011_09_26/2011_09_26_drive_0001_sync/poses.txt" ); target_size, augmentations)

# Gray.(datasets[1][1][2][:, :, 1])

# dchain = DChain(datasets)
# dataset = datasets[begin]

# Ks = []

# for d in datasets
#     push!(Ks, d.K)
# end

# Ks

# Ks[1] - mean(Ks)

# [(k ./ mean(Ks)) .- 1 for k in Ks]

# vcat([(k ./ mean(Ks)) .- 1 for k in Ks]...)

# mean(Ks)

# dl = DataLoader(dchain, 2)

# first(dl)[end].rvec

# width, height = dataset.resolution
# parameters = Params(;
#     batch_size=4, target_size=dataset.resolution,
#     disparity_smoothness=1e-3, automasking=false)
# max_scale, scale_levels = 5, collect(2:5)
# scales = [1.0 / 2.0^(max_scale - level) for level in scale_levels]
# println(parameters)

# train_cache = TrainCache(
#     transfer(SSIM()),
#     transfer(Monodepth.Backproject(; width, height)),
#     transfer(Monodepth.Project(; width, height)),
#     transfer(Array(dataset.K)), transfer(Array(dataset.invK)),
#     dataset.target_id, dataset.source_ids, scales)

# train_cache.backprojections.coordinates
# train_cache.projections.normalizer
# train_cache.K
# train_cache.K

# encoder = ResidualNetwork(18; in_channels, classes=nothing)
# encoder_channels = collect(encoder.stages)
# model = transfer(Model(
#     encoder,
#     DepthDecoder(;encoder_channels, scale_levels, embedding_levels=21),
#     PoseDecoder(encoder_channels[end])))

# θ = Flux.params(model)
# optimizer = ADAM(1e-4)
# trainmode!(model)

# # Test:

# # x = transfer(first(DataLoader(dchain, 3)))

# x = CUDA.rand(416, 128, 3, 2)

# CUDA.allowscalar(false)
# disparities = CUDA.@time model(
#     x,
#     Monodepth.uniformly_sample_disparity_from_linspace_bins(32, 2; near=1f0, far=0.001f0))



# mpi = Monodepth.network_forward(
#     model,
#     x,
#     K_inv=CUDA.rand(3, 3))

# src_img = x
# tgt_img = CUDA.rand(size(src_img)...)
# src_depth = CUDA.rand(size(src_img, 1), size(src_img, 2), 1, size(src_img, 3))



# pose = Monodepth.Pose(CUDA.rand(3, 2), CUDA.rand(3, 2))
# K = transfer(Array(dataset.K))





# invK = Monodepth.inv(K)

# Monodepth.train_loss(
#     model,
#     x,
#     src_depth,
#     tgt_img,
#     pose,
#     K,
#     invK,
#     scales)

# transfer(SSIM())(tgt_img, tgt_img)
