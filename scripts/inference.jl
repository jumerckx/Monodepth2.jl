using BSON, Images, Monodepth, Flux, CUDA, ColorSchemes
using Monodepth: Plots

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

transfer = x -> gpu(f32(x))

# BSON.@load "/scratch/vop_BC04/out/MINE/models/2-500-0.7098354643190987.bson" model_host
BSON.@load "/scratch/vop_BC04/out/MINE/models9/3-5000-3.120057814196662.bson" model_host
BSON.@load "/scratch/vop_BC04/out/MINE/models8/17-8000-3.391675782670658.bson" model_host

model = gpu(model_host)

K = train_cache.K
invK = train_cache.invK

function color_me(A, clr_map)
    n = length(clr_map)
    f(s) = clr_map[clamp(round(Int, (n-1)*s)+1, 1, n)]
    Am = map(f, A)
    return Am
end

function color_me_scaleminmax(A, cmap)
    n = length(cmap)
    scale = takemap(scaleminmax, A)
    f = s->cmap[clamp(round(Int, (n-1)*scale(s))+1, 1, n)]  # safely convert 0-1 to 1:n
    map(f, A)       # like f.(A) but does not allocate significant memory
end

dchain = DChain([datasets[2]])

disparity = nothing
src_img = nothing

imgs = []
losses = []
test = []
for (i, x_host) in enumerate(DataLoader(dchain, 1))
    if (i>10); break; end
    src_img, src_depth, tgt_img, pose = transfer.(x_host)

    println("Forward:")
    
    loss, disparity, huh = train_loss(model, src_img, src_depth, tgt_img,
        pose, K, invK, scales, N=8)
    push!(losses, huh)
    push!(test, disparity)

    disparity = collect(disparity[:, :, 1, 1])


    src_img = convert.(RGB{N0f8}, colorview(RGB, permutedims(collect(src_img[:, :, :, 1]), (3, 2, 1))))

    disparity = permutedims(disparity, (2, 1))
    disparity = color_me_scaleminmax(disparity, ColorSchemes.thermal)
    disparity = convert.(RGB{N0f8}, disparity)
    push!(imgs, vcat(src_img, disparity))
    # save("/scratch/vop_BC04/out/MINE/result_poster/$(i).png", src_img)
    # Gray.(disparity')|>display
    # fig = Plots.heatmap(
        #     disparity; aspect_ratio=:equal, xticks=nothing, yticks=nothing, colorbar=:none, legend=:none, grid=false, showaxis=false, padding = (0.0, 0.0))
        # Plots.savefig(fig, "/scratch/vop_BC04/out/MINE/result7/$i.png")
        # display(fig)
end
    

display.(imgs)

save("/scratch/vop_BC04/out/MINE/poster9.png", imgs[97])

VideoIO.save("/scratch/vop_BC04/out/MINE/video4.mp4", imgs, framerate=10, encoder_options=(crf=23, preset="medium"))

save("/scratch/vop_BC04/out/MINE/N2.png", imgs[10])

save("/scratch/vop_BC04/out/MINE/N32_disp.png", disparity)

ColorSchemes.thermal