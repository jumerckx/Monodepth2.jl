struct SupervisedDenseKITTI{A}
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
function SupervisedDenseKITTI(frames_dir, depth_dir; target_size, augmentations = nothing)

    function _get_seq_info(frames_dir, depth_dir)
        files = Set(readdir(depth_dir)) ∪ Set(readdir(depth_dir))
        ids = [parse(Int, f[1:end-4]) for f in files]
        n_frames = length(files)
        original_size = size(load(joinpath(frames_dir, first(files))))
        n_frames, ids, original_size
    end
    
    n_frames, ids, original_size = _get_seq_info(frames_dir, depth_dir)

    height, width = target_size

    SupervisedDenseKITTI(
        frames_dir,
        depth_dir,
        (width, height),
        ids,
        n_frames,
        augmentations)
end

@inline Base.length(dataset::SupervisedDenseKITTI) = dataset.total_length
function Base.getindex(d::SupervisedDenseKITTI, i)
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
end

@inline DataLoaders.nobs(dataset::SupervisedDenseKITTI) = length(dataset)
@inline DataLoaders.getobs(dataset::SupervisedDenseKITTI, i) = dataset[i]


