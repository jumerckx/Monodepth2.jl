# def get_src_xyz_from_plane_disparity(meshgrid_src_homo,

const H, W = 100, 200

invK = CUDA.rand(3, 3)


function create_meshgrid(H, W)
    CuArray(permutedims(cat((1:H) .* ones(1, W), (1:W)' .* ones(H), ones(H, W), dims=3), (3, 1, 2)))
end

@time create_meshgrid(H, W)

function uniformly_sample_disparity_from_linspace_bins(num_bins, batch_size; near=1f0, far=0.001f0)
    bin_edges_start = range(near, far, num_bins+1)[1:end-1]
    interval = bin_edges_start[2]-bin_edges_start[1]
    return bin_edges_start .+ (CUDA.rand(num_bins, batch_size)*interval)
end

N, B = 32, 4

mpi_disparity_src = uniformly_sample_disparity_from_linspace_bins(32, 4)

meshgrid_src_homo = create_meshgrid(H, W)

function get_src_xyz_from_plane_disparity(meshgrid_src_homo, mpi_disparity_src, K_src_inv)
    @show N, B = size(mpi_disparity_src)

    
    mpi_depth_src = reshape(1 ./ mpi_disparity_src, (1, 1, 1, N, B))
    @show size(mpi_depth_src)

    # return reshape(K_src_inv * reshape(meshgrid_src_homo, 3, :), (3, H, W)), mpi_depth_src

    return reshape(K_src_inv * reshape(meshgrid_src_homo, 3, :), (3, H, W)) .* mpi_depth_src
end

using LinearAlgebra
LinearAlgebra.norm(x; dims=:) = sqrt.(sum(abs2.(x), dims=dims))

xyz = get_src_xyz_from_plane_disparity(meshgrid_src_homo, mpi_disparity_src, invK)

diff = xyz[:, :, :, 2:end,:] .- xyz[:, :, :, 1:end-1, :]

dist = norm(diff, dims=1)[1, :, :, 32, 1]


norm(diff, dims=1)

using Flux


norm(diff, dims=1)

diff

plane_volume_rendering(rgb, sigma, xyz, )
