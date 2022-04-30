using BSON, Images, Monodepth, Flux, CUDA
using Monodepth: Plots

transfer = x -> gpu(f32(x))

BSON.@load "/scratch/vop_BC04/out/MINE/models/2-500-0.7098354643190987.bson" model_host

model = gpu(model_host)

src_img = load("/scratch/vop_BC04/KITTI/2011_09_28/2011_09_28_drive_0001_sync/image_02/data/0000000061.png")
src_depth = load("/scratch/vop_BC04/depth_maps/2011_09_28_drive_0001_sync/proj_depth/groundtruth/image_02/0000000061.png")

src_img = imresize(src_img, (128,416))
src_depth = imresize(src_depth, (128,416))

src_img = Flux.unsqueeze(gpu(Float32.(permutedims(channelview(src_img), (3, 2, 1)))), 4)
src_depth = gpu(Float32.(permutedims(channelview(src_depth), (2, 1))))[:, :, 1:1, 1:1]

K = gpu([ 243.98    0.0   208.0
    0.0   243.98   64.0
    0.0     0.0     1.0])
invK = inv(K)

disparities = Monodepth.uniformly_sample_disparity_from_linspace_bins(8, 1; near=eltype(src_img)(1), far=eltype(src_img)(00.001))
mpi = model(
    src_img,
    disparities;
    num_bins=8)

rgb, sigma = mpi[end]

# Monodepth.render_tgt_rgb_depth(rgb, sigma, disparities, xyz_tgt, pose, K_inv, K)

pose = Monodepth.Pose(gpu(zeros(3, 1)), gpu(zeros(3, 1)))

result = Monodepth.render_novel_view(rgb, sigma, disparities, pose, invK, K; scale=0)

Gray.(collect(result[2][:, :, 1, 1])')

save_disparity(cpu(result[2][:, :, 1, 1]))
W, H, _, N, B = size(rgb)

K = K ./ eltype(K)(2^1)
CUDA.@allowscalar K[3, 3] = 1

K_inv = inv(K)

meshgrid = Monodepth.create_meshgrid(H, W) |> transfer
xyz_src = Monodepth.get_src_xyz_from_plane_disparity(meshgrid, disparities, K_inv)

src_img_syn, src_depth_syn, blend_weights, weights = Monodepth.plane_volume_rendering(rgb, sigma, xyz_src)

save_disparity(collect(1 ./ src_depth_syn[:, :, 1, 1]), "/scratch/vop_BC04/out/MINE/img61_depth.png")

dataset

dchain = DChain([datasets[end-1]])

datasets[2]

[d.frames_dir for d in datasets]

datasets[39]

for (i, x_host) in enumerate(DataLoader(dchain, 1))
    src_img, src_depth, tgt_img, pose = transfer.(x_host)

    println("Forward:")
    loss, disparity = train_loss(model, src_img, src_depth, tgt_img,
        pose, K, invK, scales, N=16)

    disparity = collect(disparity[:, :, 1, 1])

    disparity = permutedims(disparity, (2, 1))[end:-1:1, :]
    fig = Plots.heatmap(
        disparity; aspect_ratio=:equal, xticks=nothing, yticks=nothing, colorbar=:none, legend=:none, grid=false, showaxis=false, padding = (0.0, 0.0))
    Plots.savefig(fig, "/scratch/vop_BC04/out/MINE/result/$i.png")
    display(fig)
end

disparity = rand(10, 10)
using Monodepth:Plots
Plots.heatmap(disparity, xticks=nothing, yticks=nothing, colorbar=:none, legend=:none, grid=false, showaxis=false, Plots.padding = (0.0, 0.0))