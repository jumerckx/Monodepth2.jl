using CUDA, Monodepth

CUDA.rand(3, 3) * CUDA.rand(3, 3)

W, H, N, B = 200, 100, 32, 4

src_img = CUDA.rand(W, H, 3, B)
tgt_img = CUDA.rand(W, H, 3, B)
scale = 0
K = CUDA.rand(3, 3)
mpi_rgb = CUDA.rand(W, H, 3, N, B)
mpi_sigma = CUDA.rand(W, H, 1, N, B)
disparity = Monodepth.uniformly_sample_disparity_from_linspace_bins(N, B)

pose = Monodepth.Pose(CUDA.rand(3, B), CUDA.rand(3, B))

Monodepth.loss_per_scale(src_img, tgt_img, scale, K, mpi_rgb, mpi_sigma, disparity, pose)
