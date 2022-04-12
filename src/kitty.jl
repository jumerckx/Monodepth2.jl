struct KittyDataset{A}
    frames_dir::String
    K::SMatrix{3, 3, Float64, 9}
    invK::SMatrix{3, 3, Float64, 9}
    resolution::Tuple{Int64, Int64}

    pose
    
    total_length::Int64

    augmentations::A
end

"""
- `target_size`: Size in `(height, width)` format.
"""
function KittyDataset(image_dir, calib_path, poses_path; target_size, augmentations = nothing)
    Ks = readlines(calib_path)
    K = parse_K(Ks[26][12:end])
    
    cam2_path = joinpath(image_dir, "image_02/data")
    
    n_frames, original_size = _get_seq_info(cam2_path)

    poses_to_cam0 = []
    for i in range(1, n_frames)
        Ps = readlines(poses_path)
        P = parse_P(Ps[i])
        # P = reshape(P, (4, 3))|>transpose|>collect
        P = [P; [0 0 0 1]]
        push!(poses_to_cam0, P)
    end

    poses = []
    for i in range(1, n_frames-1)
        pose_src =  poses_to_cam0[i]
        pose_dst = poses_to_cam0[i+1]
        pose_src_to_dst = (pose_dst)^(-1)*pose_src
        R_vec = try
            
            R_vec = SO3_log_map(clamp.(pose_src_to_dst[1:3, 1:3], 0, 1))
        catch e
            @show pose_src_to_dst[1:3, 1:3]
            throw(e)
        end
        t = pose_src_to_dst[1:3, 4]
        pose = Pose(R_vec, t)
        push!(poses, pose)
    end

    fx = mean(target_size ./ original_size) * K[1, 1]
    K = construct_intrinsic(fx, fx, target_size[2] ÷ 2, target_size[1] ÷ 2)
    invK = Base.inv(K)

    total_length = n_frames-1

    height, width = target_size

    KittyDataset(
        image_dir, K, invK, (width, height), poses, total_length,
        augmentations)
end

@inline Base.length(dataset::KittyDataset) = dataset.total_length
function Base.getindex(d::KittyDataset, i)
    src = load(joinpath(d.frames_dir, "image_02/data", "$(lpad(i, 10, "0")).png" ))
    tgt = load(joinpath(d.frames_dir, "image_02/data", "$(lpad(i+1, 10, "0")).png" ))
    
    width, height = d.resolution
    src, tgt = map(x -> imresize(x, (height, width)), (src, tgt))
    if d.augmentations ≢ nothing
        (src,tgt) = d.augmentations((src,tgt))
    end
    src,tgt = map(
        x -> permutedims(Float32.(channelview(x)), (3, 2, 1)),
        (src,tgt))
    return src,tgt, d.pose[i]
    # cat(images...; dims=4)
end

@inline DataLoaders.nobs(dataset::KittyDataset) = length(dataset)
@inline DataLoaders.getobs(dataset::KittyDataset, i) = dataset[i]

function _get_seq_info(seq_dir::String)
    files = readdir(seq_dir; sort=false)
    n_frames = length(files)
    original_size = size(load(joinpath(seq_dir, files[begin])))
    n_frames, original_size
end

function parse_K(line)
    m = parse.(Float64, split(line, " "))
    K = SMatrix{4, 4, Float64}(m..., 0, 0, 0, 1)'
    K[1:3, 1:3]
end
function parse_R(line)
    m = parse.(Float64, split(line, " "))
    R = SMatrix{3, 3, Float64}(m...)'
    return R
end
function parse_T(line)
    m = parse.(Float64, split(line, " "))
    T = SVector{3, Float64}(m...)
    return T
end

function parse_P(line)
    m = parse.(Float64, split(line, " "))
    P = SMatrix{4,3, Float64, 12}(m...)'
    return P
end

@inline function construct_intrinsic(fx, fy, cx, cy)
    SMatrix{3, 3, Float64, 9}(
        fx, 0, 0,
        0, fy, 0,
        cx, cy, 1)
end
