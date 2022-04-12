using Images, CUDA

include("bilateral_guided_filtering.jl")

img_root = "/scratch/vop_BC04/KITTI/"
src_depth_root = "/home/lab/Documents/vop_BC04/untitled folder 2/train/"
target_depth_root = "/scratch/vop_BC04/depth_maps"

function bilateral_guided_depth_completion(depth_path, img_path)
    depth = cu(Float32.(channelview(load(depth_path))))
    img = cu(sum(Float32.(channelview(load(img_path))) .* [0.2989, 0.5870, 0.1140], dims=1))

    bilateral_guided_depth_completion(depth, img, device=CUDADevice, S=(3, 3), iters=3)
end

function convert(target_depth_root, src_depth_root, img_root)
    i = 0
    for (root, dirs, files) in walkdir(img_root)
        for file in files
            if i > 50
                println("einde")
                return
            end
            extension = splitext(file)[2]
            if extension == ".png"
                # println(root)
                # println(file)
                sp = splitpath(root)
                # @show sp
                src_depth = joinpath(src_depth_root, sp[6], "proj_depth", "groundtruth", sp[7:end-1]..., file)
                # @show src_depth
                if isfile(src_depth)
                    println("test")
                    Gray.(collect(bilateral_guided_depth_completion(src_depth, joinpath(root, file))))|>display
                    i += 1
                end
            end
        end
    end
end

convert(target_depth_root, src_depth_root, img_root)


