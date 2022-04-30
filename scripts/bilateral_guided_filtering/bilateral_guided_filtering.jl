using CUDA, KernelAbstractions, Images, OffsetArrays, CUDAKernels, Images, StaticArrays, BenchmarkTools

@kernel function bilateral_guided_depth_completion_kernel(depth_out, @Const(depth_in), @Const(img), ::Val{S}, h=(0.25, 0.1, 0.1)) where S
	i, j = @index(Global, NTuple)
	li, lj = @index(Local, NTuple)
	
	@uniform gs = @groupsize()	
	
	@synchronize()

	lmem_img = @localmem eltype(img) (gs .+ (2 .* S))
	lmem_depth = @localmem eltype(depth_in) (gs .+ ( 2 .* S))

	@uniform local_img = OffsetArray(lmem_img,  1-S[1]:gs[1]+S[1], 1-S[2]:gs[2]+S[2])
	@uniform local_depth = OffsetArray(lmem_depth, 1-S[1]:gs[1]+S[1], 1-S[2]:gs[2]+S[2])

	@inbounds begin
		local_img[li, lj] = img[i, j]
		local_depth[li, lj] = depth_in[i, j]

		if li <= S[1] #top
			local_img[-S[1]+li, lj] = img[-S[1]+i, j]
			local_depth[-S[1]+li, lj] = depth_in[-S[1]+i, j]
		end
		if li <= S[1] && lj <= S[2] #top-left
			local_img[-S[1]+li, -S[2]+lj] = img[-S[1]+i, -S[2]+j]
			local_depth[-S[1]+li, -S[2]+lj] = depth_in[-S[1]+i, -S[2]+j]
		end
		if lj <= S[2] #left
			local_img[li, -S[2]+lj] = img[i, -S[2]+j]
			local_depth[li, -S[2]+lj] = depth_in[i, -S[2]+j]
		end
		if lj <= S[2] && li >= gs[1] - S[1] #bottom-left
			local_img[li+S[1], -S[2]+lj] = img[i+S[1], -S[2]+j]
			local_depth[li+S[1], -S[2]+lj] = depth_in[i+S[1], -S[2]+j]
		end
		if li >= gs[1] - S[1] #bottom
			local_img[li+S[1], lj] = img[i+S[1], j]
			local_depth[li+S[1], lj] = depth_in[i+S[1], j]
		end
		if li >= gs[1] - S[1] && lj >= gs[2] - S[2] #bottom-right
			local_img[li+S[1], lj+S[2]] = img[i+S[1], j+S[2]]
			local_depth[li+S[1], lj+S[2]] = depth_in[i+S[1], j+S[2]]
		end
		if lj >= gs[2] - S[2] #right
			local_img[li, lj+S[2]] = img[i, j+S[2]]
			local_depth[li, lj+S[2]] = depth_in[i, j+S[2]]
		end
		if lj >= gs[2] - S[2] && li <= S[1] #top-right
			local_img[-S[1]+li, lj+S[2]] = img[-S[1]+i, j+S[2]]
			local_depth[-S[1]+li, lj+S[2]] = depth_in[-S[1]+i, j+S[2]]
		end
	end

	@synchronize()
	
	dist = typemax(eltype(depth_out))
	nearest_val = 0

	for k in -S[1]:S[1]
		for l in -S[2]:S[2]
			if (depth_in[i+k, j+l] > 0) && ( √(k^2+l^2) < dist)
				dist = √(k^2 + l^2)
				nearest_val = local_depth[li+k, lj+l]
			end
		end
	end
		
	depth_val = 0
	final_weight = 0
	total_weight = 0

	for k in -S[1]:S[1]
		for l in -S[2]:S[2]
			if (depth_in[i+k, j+l] > 0)
				weight = exp(-(
					h[1] * (local_img[li, lj] - local_img[li+k, lj+l])^2
					+ h[2] * √(k^2+l^2)
					+ h[3] * (nearest_val - local_depth[li+k, lj+l])^2))

				total_weight += weight
				depth_val += weight * local_depth[li+k, lj+l]
			end
		end
	end

	if total_weight > 0
		depth_out[i, j] = depth_val/total_weight
		# depth_weight[m, n] = total_weight
	end
end

function bilateral_guided_depth_completion(depth, img; iters=1, device=CPU, S=(3, 3), workgroupsize=(8, 8))
	H, W = size(depth)
	
	depth_in = zero.(similar(depth, size(depth) .+ (2 .* S)))
	depth_in[(1:H) .+ S[1], (1:W) .+ S[2]] .= depth
	
	depth_in = OffsetArray(depth_in, 1-S[1]:H+S[1], 1-S[2]:W+S[2]) # zero padding
	depth_out = deepcopy(depth_in)

	k = bilateral_guided_depth_completion_kernel(device(), workgroupsize)
	for i in 1:iters
		depth_in, depth_out = depth_out, depth_in

		wait(k(depth_out, depth_in, img, Val(S); ndrange=size(depth)))

	end
	return depth_out.parent[(1:H) .+ S[1], (1:W) .+ S[2]]
end

begin # testafbeeldingen inladen.
	img = Float64.(channelview(load("./scripts/bilateral_guided_filtering/img.png")))
	# Gray.(img)|>display
	sparse_depth = Float64.(channelview(load("./scripts/bilateral_guided_filtering/depth.png")))
	# Gray.(sparse_depth)|>display
	H, W = size(img)
end

# test op gpu (CUDADevice):
sparse_depth, img = cu(sparse_depth), cu(img)
sparse_depth = CUDA.rand(100, 100) .* (CUDA.rand(100, 100) .> 0.5)
img = CUDA.rand(100, 100)
@time for _ in 1:100
	CUDA.@sync bilateral_guided_depth_completion(sparse_depth, img, S=(3, 3), iters=1,
	device=CUDADevice, workgroupsize=(8, 8))
end

CUDA.@time begin
	temp = bilateral_guided_depth_completion(cu(sparse_depth), cu(img), S=(3, 3), iters=2, device=CUDADevice) # cu plaatst array op gpu.
	temp = collect(temp) # terug op CPU plaatsen.
	Gray.(temp)|>display # Float omzetten naar type Gray zodat afbeelding kan worden weergegeven.
end

# # test op CPU:
# Gray.(bilateral_guided_depth_completion(sparse_depth, img, S=(1, 1), iters=2, device=CPU))


# f = bilateral_guided_depth_completion_kernel(CPU(), (8, 8))

# begin
# 	S = (5, 5)
# 	img_in = OffsetArray(img, -S[1]+1:H-S[1], -S[2]+1:W-S[2])
# 	depth_in = OffsetArray(sparse_depth, -S[1]+1:H-S[1], -S[2]+1:W-S[2])
# 	depth_out = deepcopy(depth_in)
# end;


# begin
# 	e = f(depth_out, depth_in, img_in, Val(S); ndrange=(H-2*S[1], W-2*S[2]))
# 	wait(e)
	
# 	Gray.(depth_out)
# end

# g = bilateral_guided_depth_completion_kernel(CUDADevice(), (8, 8))

# begin
# 	CUDA.allowscalar(true)
# 	img_in_cu = OffsetArray(CuArray(img), -S[1]+1:H-S[1], -S[2]+1:W-S[2])
# 	depth_in_cu = OffsetArray(CuArray(sparse_depth), -S[1]+1:H-S[1], -S[2]+1:W-S[2])
# 	depth_out_cu = deepcopy(depth_in_cu)
# 	CUDA.allowscalar(false)
# end

# begin
# 	# wait(g(depth_out_cu, depth_in_cu, img_in_cu, Val(S), ndrange=(H-2*S[1], W-2*S[2])))
# 	@btime CUDA.@sync wait(g(depth_out_cu, depth_in_cu, img_in_cu, Val(S), ndrange=(H-2*S[1], W-2*S[2])))
# 	Gray.(collect(depth_out_cu.parent))
# end