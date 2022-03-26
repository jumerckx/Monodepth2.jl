# deze file is enkel nog ter referentie

using Flux: unsqueeze, grid_sample

Flux.unsqueeze

W, H = 384, 256

pose = poses[1]

R = CuArray(I(3))
t = pose.tvec[:, :, 1:1]
t .= CuArray([0, 0, -0.09])


n = transfer([0 0 1])
depth_src = repeat([1.0020,  1.0595,  1.0762,  1.1304,  1.1716,  1.2110,  1.2542,  1.2882,
1.3659,  1.4097,  1.5151,  1.5346,  1.6515,  1.7619,  1.8006,  1.9431,
2.0716,  2.1703,  2.4325,  2.5377,  2.7491,  3.1771,  3.2049,  3.8405,
3.9952,  4.8218,  5.8518,  6.6441, 10.4757, 11.9267, 20.6962, 34.5137], 1, 1)|>transfer

K = [
    192.   0. 192.
    0. 192. 128.
    0.   0.   1.
]|>transfer
K_inv = inv(K)

t = Flux.unsqueeze(pose.tvec, 2)
R = so3_exp_map(pose.rvec)
temp = Flux.unsqueeze(t ⊠ n, 3) ./ -reshape(depth_src, (1, 1, size(depth_src, 1), size(depth_src, 2)))
H_tgt_src = K ⊠ (reshape(Flux.unsqueeze(R, 3) .- temp, (3, 3, :))) ⊠ K_inv


H_src_tgt = inv(H_tgt_src)

meshgrid_tgt_homo = transfer(reshape(create_meshgrid(H, W), (3, :)) .- [1, 1, 0])
meshgrid_src = H_src_tgt ⊠ meshgrid_tgt_homo

meshgrid_src = meshgrid_src[1:2, :, :] ./ meshgrid_src[3:3, :, :]

valid_mask = (meshgrid_src[1, :, :] .< W .* meshgrid_src[1, :, :] .>= 0) .* (meshgrid_src[2, :, :] .< H .* meshgrid_src[2, :, :] .>= 0)

meshgrid_src[1, :, :] .= (meshgrid_src[1, :, :] .+ eltype(meshgrid_src)(0.5)) ./ (W/2)
meshgrid_src[2, :, :] .= (meshgrid_src[2, :, :] .+ eltype(meshgrid_src)(0.5)) ./ (H/2)

meshgrid_src = reshape(meshgrid_src, (2, W, H, :))

src = CUDA.rand(W, H, 7, 32*2)

grid_sample(src, meshgrid_src; padding_mode=:border)




# ----------------- inv ---------------------
function inv!(x)
    pivot, info = CUDA.CUBLAS.getrf_batched!(x, true)
    _,_,x = CUDA.CUBLAS.getri_batched(x, pivot)
    return x
end
function inv!(x::CuArray{T, 2}) where T
    return inv!([x])[1]
end
function inv!(x::CuArray{T, 3}) where T
    cat(inv!(collect(eachslice(x, dims=3)))..., dims=3) # TODO: moet sneller kunnen
end
import LinearAlgebra:inv
function inv(x::CuArray{T, N}) where {T, N}
    x = deepcopy(x)
    return inv!(x)
end
