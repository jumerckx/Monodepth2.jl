using PyCall, CUDA, Flux

function plane_volume_rendering(rgb, sigma, xyz)
    diff = permutedims(xyz[:, :, :, 2:end,:] .- xyz[:, :, :, 1:end-1, :], (2, 3, 1, 4, 5))
    dist = cat(norm(diff, dims=3), CUDA.fill(1f3, W, H, 1, 1, B), dims=4) # TODO: TinyNERF gebruikt fill met 1e10

    transparency = exp.(-dist .* sigma)
    alpha = 1 .- transparency

    transparency_acc = cumprod(transparency .+ 1e-6, dims=4) # TODO: is ".+ 1e-6 " nodig?
    transparency_acc = cat(CUDA.ones(W, H, 1, 1, B), transparency_acc[:, :, :, 1:end-1, :], dims=4)

    weights = transparency_acc .* alpha  # BxSx1xHxW

    rgb_out = dropdims(sum(weights .* rgb, dims=4), dims=4)

    # TODO: return depth?
    return rgb_out, transparency_acc, weights
end

push!(pyimport("sys")."path", "/mnt/c/Users/jules/OneDrive - UGent/Documenten/code/VOP/MINE/MINE")

@pyimport operations.mpi_rendering as mpi_rendering
@pyimport torch as torch
@pyimport numpy as np

B, S, H, W = 4, 32, 100, 200

cuda0 = torch.device("cuda:0")

rgb_BS3HW, sigma_BS1HW, xyz_BS3HW = rand(B, S, 3, H, W), rand(B, S, 1, H, W), rand(B, S, 3, H, W)

rgb = permutedims(rgb_BS3HW, (5, 4, 3, 2, 1))|>gpu
sigma = permutedims(sigma_BS1HW, (5, 4, 3, 2, 1))|>gpu
xyz = permutedims(xyz_BS3HW, (3, 5, 4, 2, 1))|>gpu

rgb_BS3HW, sigma_BS1HW, xyz_BS3HW = [torch.tensor(x, device=cuda0) for x in (rgb_BS3HW, sigma_BS1HW, xyz_BS3HW)]

norm(x; dims=:) = sqrt.(sum(abs2.(x), dims=dims))

rgb_out, depth_out, transparency_acc, weights = mpi_rendering.plane_volume_rendering(rgb_BS3HW, sigma_BS1HW, xyz_BS3HW, false)
rgb_out2, transparency_acc2, weights2 = plane_volume_rendering(rgb, sigma, xyz)

@show rgb_out.shape

@show size(rgb_out2)

isapprox(collect(permutedims(rgb_out2, (4, 3, 2, 1))), np.array(rgb_out.cpu()), rtol=1e-7)

# py"""
# import torch
# from operations import mpi_rendering
# xyz_diff_BS3HW = $xyz_BS3HW[:, 1:, :, :, :] - $xyz_BS3HW[:, 0:-1, :, :, :]  # Bx(S-1)x3xHxW
# xyz_dist_BS1HW = torch.norm(xyz_diff_BS3HW, dim=2, keepdim=True)  # Bx(S-1)x1xHxW

# xyz_dist_BS1HW = torch.cat((xyz_dist_BS1HW,
#                             torch.full(($B, 1, 1, $H, $W),
#                                         fill_value=1e3,
#                                         dtype=$xyz_BS3HW.dtype,
#                                         device=$xyz_BS3HW.device)),
#                             dim=1)  # BxSx3xHxW
# transparency = torch.exp(-$sigma_BS1HW * xyz_dist_BS1HW)  # BxSx1xHxW
# alpha = 1 - transparency # BxSx1xHxW
# transparency_acc = torch.cumprod(transparency + 1e-6, dim=1)  # BxSx1xHxW
# transparency_acc = torch.cat((torch.ones(($B, 1, 1, $H, $W), dtype=transparency.dtype, device=transparency.device),
#                                 transparency_acc[:, 0:-1, :, :, :]),
#                                 dim=1)  # BxSx1xHxW
# weights = transparency_acc * alpha  # BxSx1xHxW
# rgb_out, depth_out = mpi_rendering.weighted_sum_mpi($rgb_BS3HW, $xyz_BS3HW, weights, False)
                            
# """


# diff = permutedims(xyz[:, :, :, 2:end,:] .- xyz[:, :, :, 1:end-1, :], (2, 3, 1, 4, 5))
# dist = cat(norm(diff, dims=3), CUDA.fill(1f3, W, H, 1, 1, B), dims=4) # TODO: TinyNERF gebruikt fill met 1e10
# transparency = exp.(-dist .* sigma)
# alpha = 1 .- transparency
# transparency_acc = cumprod(transparency .+ 1e-6, dims=4) # TODO: is ".+ 1e-6 " nodig?
# transparency_acc = cat(CUDA.ones(W, H, 1, 1, B), transparency_acc[:, :, :, 1:end-1, :], dims=4)
# # transparency_acc = permutedims(np.array(py"transparency_acc".cpu()), (5, 4, 3, 2, 1))|>gpu
# weights = transparency_acc .* alpha  # BxSx1xHxW
# rgb_out = dropdims(sum(weights .* rgb, dims=4), dims=4)

# # 200,100,1,4