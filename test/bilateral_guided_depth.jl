using Images

rgb = permutedims(channelview(load("/scratch/vop_BC04/KITTI/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png")), (3, 2, 1))
depth = Float64.(permutedims(channelview(load("/home/lab/Documents/vop_BC04/untitled folder 2/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png")), (2, 1))[:, :, 1:1])

function bilateral_guided_depth_completion(img_in, depth_in, num_iterations)
    S = (11, 13)
    depth_out = zeros(size(depth_in))
    depth_weight = zeros(size(depth_in))
    h1 = 0.25
    h2 = 0.1
    h3 = 0.1
    thresh = 0.01

    for it in 1:num_iterations
        println("ITERATIE")
        if it > 1                 # buffer swap
            depth_in,depth_out = depth_out,depth_in
        end
        for m in 1:size(img_in, 1)
            for n in 1:size(img_in, 2)
                depth_val = 0.0
                total_weight = 0.0
                final_weight = 0.0
                ref_val = img_in[m,n,:]
                dist = typemax(eltype(depth_in))
                nearest_val = 0.0
                for k in -S[1]:S[1]
                    for l in -S[2]:S[2]
                        if (1 <= m+k <= size(img_in, 1))  && (1 <= n+l <= size(img_in, 2))
                            if depth_in[m+k,n+l]>0
                                if sqrt(k^2+l^2)<dist
                                    dist = sqrt(k^2+l^2)
                                    nearest_val = depth_in[m+k,n+l]
                                end
                            end
                        end
                    end
                end
                for k in -S[1]:S[1]
                    for l in -S[2]:S[2]
                        if (1 <= m+k <= size(img_in, 1))  && (1 <= n+l <= size(img_in, 2))
                            if depth_in[m+k,n+l]>0
                                h = (
                                    h1 * sum((ref_val.-img_in[m+k,n+l,:]).^2),          # Intensity
                                    h2 * sqrt(k^2+l^2),                                 # Space
                                    h3 * (nearest_val-depth_in[m+k,n+l])^2              # Depth
                                    )             
                                weight = exp(-sum(h))
                                total_weight += weight
                                depth_val += weight*depth_in[m+k,n+l]
                            end
                        end
                    end
                end
                if total_weight>0
                    depth_out[m,n] = depth_val/total_weight
                    depth_weight[m,n] =  total_weight
                end # Otherwise, needs to be estimated in the next iteration
            end
        end
    end    



    return depth_out, depth_weight
end

depth_out, depth_weight = bilateral_guided_depth_completion(rgb, depth, 3)

@code_warntype bilateral_guided_depth_completion(rgb, depth, 3)

Gray.(permutedims(depth_out[:, :, 1], (2, 1)))
