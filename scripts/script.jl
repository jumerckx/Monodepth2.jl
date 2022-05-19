using LinearAlgebra
using Printf
using Statistics
using Images

using Monodepth, Augmentations
using BSON: @save, @load

using DataLoaders
using ProgressMeter
using CUDA
using Flux
using Monodepth.ResNet

function train(; η=1e-4, model=nothing, θ=nothing, datasets=nothing)
    device = gpu
    precision = f32
    transfer = device ∘ precision
    @show transfer

    log_dir = "/scratch/vop_BC04/out/MINE/logs9"
    save_dir = "/scratch/vop_BC04/out/MINE/models9"

    isdir(log_dir) || mkpath(log_dir)
    isdir(save_dir) || mkpath(save_dir)

    grayscale = false
    in_channels = grayscale ? 1 : 3
    augmentations = FlipX(0.5)


    target_size= (128,416)
    # target_size = (64, 64)


    
    if isnothing(datasets)
        img_dir = "/scratch/vop_BC04/KITTI/"
        depth_dir = "/scratch/vop_BC04/depth_maps/"

        datasets = []
        for datum in filter(isdir, joinpath.(img_dir, readdir(img_dir)))
            datum_path = joinpath(img_dir, datum)
            calib_path = joinpath(datum_path, "calib_cam_to_cam.txt")
            for drive in readdir(datum_path)[isdir.(joinpath.(datum_path, readdir(datum_path)))] # filter(isdir, joinpath.(datum_path,  readdir(datum_path)))
                drive_depth_path = joinpath(depth_dir, drive, "proj_depth", "groundtruth")
                if (isdir(drive_depth_path))
                    drive = joinpath(datum_path, drive)
                    poses_path = joinpath(drive, "poses.txt")
                    push!(datasets, KittyDataset(drive, drive_depth_path, calib_path, poses_path; target_size, augmentations))
                end
            end
        end
    end

    dchain = DChain(datasets)
    dataset = datasets[begin]

    width, height = dataset.resolution
    parameters = Params(;
        batch_size=1, target_size=dataset.resolution,
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
            DepthDecoder(; encoder_channels, scale_levels, embedding_levels=21),
            PoseDecoder(encoder_channels[end])))
    end
    if (isnothing(θ))
        θ = Flux.params(model)
    end

    optimizer = ADAM(η)
    trainmode!(model)

    # Perform first gradient computation using small batch size.
    println("Precompile grads...")
    for x_host in DataLoader(dchain, 1)
        src_img, src_depth, tgt_img, pose = transfer.(x_host)

        println("Forward:")
        @time train_loss(model, src_img, src_depth, tgt_img,
            pose, train_cache.K, train_cache.invK, scales, N=2)


        println("Backward:")
        @time begin
            ∇ = gradient(θ) do
                loss = train_loss(model, src_img, src_depth, tgt_img,
                    pose, train_cache.K, train_cache.invK, scales, N=2)[1]
            end
        end

        # @show mean(∇[model.pose_decoder.pose[end].weight])
        break
    end
    GC.gc()

    # Do regular training.
    n_epochs, log_iter, save_iter = 20, 50, 1000

    println("Training...")
    for epoch in 1:n_epochs
        loader = DataLoader(DataLoaders.shuffleobs(dchain), parameters.batch_size)
        bar = Monodepth.get_pb(length(loader), "Epoch $epoch / $n_epochs: ")

        for (i, x_host) in enumerate(loader)
            src_img, src_depth, tgt_img, pose = transfer.(x_host)
            auto_loss = nothing
            # if parameters.automasking
            #     auto_loss = automasking_loss(
            #         train_cache.ssim, x, x[:, :, :, train_cache.target_id, :];
            #         source_ids=train_cache.source_ids)
            # end

            loss_cpu = 0.0
            losses_cpu = 0.0
            disparity, warped, vis_loss = nothing, nothing, nothing
            do_visualization = i % log_iter == 0 || i == 1


            Flux.Optimise.update!(optimizer, θ, gradient(θ) do
                loss, disparity, losses = train_loss(
                    model, src_img, src_depth, tgt_img,
                    pose, train_cache.K, train_cache.invK, scales, N=8)
                # loss = train_loss(
                #     model, src_img, src_depth, tgt_img,
                #     pose, train_cache.K, train_cache.invK, scales, N=2)
                loss_cpu = cpu(loss)
                losses_cpu = cpu(losses)
                loss
            end)

            if do_visualization
                temp = cpu(disparity[:, :, 1, 1])
                # save_disparity(temp)
                # colorview(RGB, permutedims(collect(src_img[:, :, :, 1]), (3, 2, 1))) |> display

                save_disparity(
                    temp,
                    joinpath(log_dir, "loss-$epoch-$i.png"))
            end
            if i % save_iter == 0
                model_host = cpu(model)
                @save joinpath(save_dir, "$epoch-$i-$loss_cpu.bson") model_host
            end

            next!(bar; showvalues=[(:i, i), (:loss, losses_cpu)])
        end
    end
end
CUDA.allowscalar(false)
train(datasets=datasets[1:48])

# testset is datasets[128:129]

device = gpu
precision = f32
transfer = device ∘ precision
@show transfer

in_channels = 3
# augmentations = FlipX(0.5)
augmentations = nothing

target_size= (128,416)
# target_size = (64, 64)


img_dir = "/scratch/vop_BC04/KITTI/"
depth_dir = "/scratch/vop_BC04/depth_maps/"

datasets = []
for datum in filter(isdir, joinpath.(img_dir, readdir(img_dir)))
    datum_path = joinpath(img_dir, datum)
    calib_path = joinpath(datum_path, "calib_cam_to_cam.txt")
    for drive in readdir(datum_path)[isdir.(joinpath.(datum_path, readdir(datum_path)))] # filter(isdir, joinpath.(datum_path,  readdir(datum_path)))
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

encoder = ResidualNetwork(18; in_channels, classes=nothing)
encoder_channels = collect(encoder.stages)
model = transfer(Model(
    encoder,
    DepthDecoder(; encoder_channels, scale_levels, embedding_levels=21),
    PoseDecoder(encoder_channels[end])))
θ = Flux.params(model)

x_host = first(DataLoader(DataLoaders.shuffleobs(dchain), 4))
src_img, src_depth, tgt_img, pose = transfer.(x_host)

disparities = Monodepth.uniformly_sample_disparity_from_linspace_bins(8, 4; near=eltype(src_img)(1), far=eltype(src_img)(0.001))
mpi_rgb, mpi_sigma = model(
    src_img,
    disparities;
    num_bins=8)[end]

src_disparity_syn, loss_rgb_tgt, loss_ssim_tgt, loss_smooth_src, loss_smooth_tgt, loss_depth = Monodepth.loss_per_scale(src_img, src_depth, tgt_img, scales[end], train_cache.K, mpi_rgb, mpi_sigma, disparities, pose)

f() = begin
    for i in 1:100
        encoder = ResidualNetwork(18; in_channels, classes=nothing)
        encoder_channels = collect(encoder.stages)
        model = transfer(Model(
            encoder,
            DepthDecoder(; encoder_channels, scale_levels, embedding_levels=21),
            PoseDecoder(encoder_channels[end])))
        θ = Flux.params(model)

        trainmode!(model)

        # Perform first gradient computation using small batch size.
        x_host = first(DataLoader(DataLoaders.shuffleobs(dchain), 1))
        src_img, src_depth, tgt_img, pose = transfer.(x_host)

        out = train_loss(model, src_img, src_depth, tgt_img, pose, train_cache.K, train_cache.invK, scales, N=2)
    end
end