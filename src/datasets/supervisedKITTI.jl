struct SupervisedKITTI{A}
    frames_dir::String
    depth_dir::String

    # K::SMatrix{3, 3, Float64, 9}
    # invK::SMatrix{3, 3, Float64, 9}
    resolution::Tuple{Int64, Int64}

    ids::Vector{Int64}
    # target_id::Int64
    # frame_ids::Vector{Int64}
    total_length::Int64

    augmentations::A
end

"""
- `target_size`: Size in `(height, width)` format.
"""
function SupervisedKITTI(frames_dir, depth_dir; target_size, augmentations = nothing)

    function _get_seq_info(frames_dir, depth_dir)
        files = Set(readdir(depth_dir; sort=false)) ∪ Set(readdir(depth_dir))
        ids = [parse(Int, f[1:end-4]) for f in files]
        n_frames = length(files)
        original_size = size(load(joinpath(frames_dir, first(files))))
        n_frames, ids, original_size
    end

    frames_dir = joinpath(frames_dir, "image_02", "data")
    depth_dir = joinpath(depth_dir, "proj_depth", "groundtruth", "image_02")
    
    n_frames, ids, original_size = _get_seq_info(frames_dir, depth_dir)

    height, width = target_size

    SupervisedKITTI(
        frames_dir,
        depth_dir,
        (width, height),
        ids,
        n_frames,
        augmentations)
end

@inline Base.length(dataset::SupervisedKITTI) = dataset.total_length
function Base.getindex(d::SupervisedKITTI, i)
    rgb_fname, depth_fname = joinpath.((d.frames_dir, d.depth_dir), @sprintf("%.010d.png", d.ids[i]))
    width, height = d.resolution
    rgb, depth = imresize(load(rgb_fname), (height, width)), imresize(load(depth_fname), (height, width))
    if d.augmentations ≢ nothing
        rgb, depth = d.augmentations((rgb, depth))
    end
    rgb, depth = channelview.((rgb, depth))

    # return rgb, depth
    rgb = Float32.(permutedims(rgb, (3, 2, 1)))
    depth = Float32.(Flux.unsqueeze(permutedims(depth, (2, 1)), 3))

    return rgb, depth


    # sid = (i - 1) * length(d.frame_ids)
    # images = map(
    #     x -> load(joinpath(d.frames_dir, @sprintf("%.06d.png", sid + x - 1))),
    #     d.frame_ids)

    # width, height = d.resolution
    # images = map(x -> imresize(x, (height, width)), images)
    # if d.augmentations ≢ nothing
    #     images = d.augmentations(images)
    # end

    # images = map(
    #     x -> Flux.unsqueeze(permutedims(Float32.(channelview(x)), (2, 1)), 3),
    #     images)
    # cat(images...; dims=4)
end

@inline DataLoaders.nobs(dataset::SupervisedKITTI) = length(dataset)
@inline DataLoaders.getobs(dataset::SupervisedKITTI, i) = dataset[i]


