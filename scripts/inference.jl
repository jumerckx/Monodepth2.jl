using BSON, Images, Monodepth, Flux, CUDA, ColorSchemes
using Monodepth: Plots

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