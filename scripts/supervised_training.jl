using LinearAlgebra
using Printf
using Statistics

using Monodepth, Augmentations
using BSON: load, @save, @load
using DataLoaders
# using FileIO
# using ImageCore
# using ImageTransformations
using Images
using Monodepth: shuffleobs, get_pb, next!
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

# img_dir = "../nyu_data/data/nyu2_train"
# datasets = [
#     NYUDataset(joinpath(img_dir, dir), (240,320); augmentations=nothing)
#     for dir in readdir(img_dir)
# ]

# dchain = DChain(datasets)

# length(dchain)

# # dchain[1][1]



# img_dir = "/scratch/vop_BC04/KITTI/2011_09_26"
# depth_dir = "../untitled folder 2/train/"

# dataset = [
#     SupervisedKITTI(
#         joinpath(img_dir, dir),
#         joinpath(depth_dir, dir);
#         target_size=(128,416),
#         augmentations=nothing)
#     for dir in Set(readdir(img_dir)) ∩ Set(readdir(depth_dir))
#     ][2]



# using Images
# rgb = permutedims(dataset[3][1], (3, 2, 1))
# gray = permutedims(dataset[3][2][:, :, 1], (2, 1))
# colorview(Gray, gray ./ maximum(gray))
# colorview(RGB, rgb,)

# count(gray .!= 0)/length(gray)

# gray[gray .!= 0]

# maximum(permutedims(first(datasets)[1][2][:, :, 1], (2, 1)))

# readdir(img_dir)

# disparities = model(gpu(dataset[3][1][:, :, :, 1:1]))


# Gray.(collect(disparities[4][:, :, 1, 1])')

img_dir = "/scratch/depth_completion/depth_selection/test_depth_completion_anonymous/image/"
depth_dir = "/scratch/depth_completion/depth_selection/test_depth_completion_anonymous/output_png/"

dataset = Monodepth.SupervisedDenseKITTI(
        img_dir,
        depth_dir;
        target_size=(128,416),
        augmentations=nothing)

dataset.ids

permutedims(dataset[4][2][:, :, 1], (2, 1))

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
    
    # target_size = (384,512)
    # img_dir = "../nyu_data/data/nyu2_train"
    # datasets = [
    # NYUDataset(joinpath(img_dir, dir), target_size; augmentations=nothing)
    # for dir in readdir(img_dir)
    # ]
    
    target_size= (128,416)
    # img_dir = "/scratch/vop_BC04/KITTI/2011_09_26"
    # depth_dir = "../untitled folder 2/train/"

    # datasets = [
    #     SupervisedKITTI(
    #         joinpath(img_dir, dir),
    #         joinpath(depth_dir, dir);
    #         target_size,
    #         augmentations)

    #     for dir in Set(readdir(img_dir)) ∩ Set(readdir(depth_dir))
    # ]

    img_dir = "/scratch/depth_completion/depth_selection/test_depth_completion_anonymous/image/"
    depth_dir = "/scratch/depth_completion/depth_selection/test_depth_completion_anonymous/output_png/"

    datasets = [SupervisedDenseKITTI(
            img_dir,
            depth_dir;
            target_size,
            augmentations)]

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
        scales)
    
    if (isnothing(model))
        encoder = ResidualNetwork(18; in_channels, classes=nothing)
        encoder_channels = collect(encoder.stages)
        model = transfer(Model(
            encoder,
            DepthDecoder(;encoder_channels, scale_levels, embedding_levels=0),
            PoseDecoder(encoder_channels[end])))
    end
    if (isnothing(θ)); θ = Flux.params(model); end

    optimizer = ADAM(η)
    trainmode!(model)

    # Perform first gradient computation using small batch size.
    println("Precompile grads...")
    for x_host in DataLoader(dchain, 1)
        x, y = transfer.(x_host)

        println("Forward:")
        @time train_loss(model, x, y, train_cache, parameters, false)[1]

        println("Backward:")
        @time begin
            ∇ = gradient(θ) do
                train_loss(model, x, y, train_cache, parameters, false)[1]
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
            x, y = transfer.(x_host)

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
                    model, x, y, train_cache,
                    parameters, do_visualization)
                loss_cpu = cpu(loss)
                loss
            end)

            if do_visualization
                save_disparity(disparity[:, :, 1, 1])
                colorview(RGB, permutedims(collect(x[:, :, :, 1]), (3, 2, 1)))|>display

                save_disparity(
                    disparity[:, :, 1, 1],
                    joinpath(log_dir, "loss-$epoch-$i.png"))
            end
            if i % save_iter == 0
                model_host = cpu(model)
                @save joinpath(save_dir, "$epoch-$i-$loss_cpu.bson") model_host
            end

            next!(bar; showvalues=[(:i, i), (:loss, loss_cpu)])
        end
    end
end

# model = gpu(f32(load("/scratch/vop_BC04/out/models/4-500-1.3209074.bson")[:model_host]))

train()
