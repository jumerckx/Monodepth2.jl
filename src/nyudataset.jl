struct NYUDataset{A}
    image_dir::String
    resolution::Tuple{Int64, Int64}
    total_length::Int64
    augmentations::A
end

"""
- `target_size`: Size in `(height, width)` format.
"""
function NYUDataset(image_dir, target_size; augmentations = nothing)

    total_length = length(readdir(image_dir)) ÷ 2

    height, width = target_size

    NYUDataset(
        image_dir,
        (width, height),
        total_length,
        augmentations)
end

@inline Base.length(dataset::NYUDataset) = dataset.total_length
function Base.getindex(d::NYUDataset, i)
    rgb_fname = joinpath(d.image_dir, "$i.jpg")
    depth_fname = joinpath(d.image_dir, "$i.png")
    width, height = d.resolution
    rgb, depth = imresize(load(rgb_fname), (height, width)), imresize(load(depth_fname), (height, width))
    if d.augmentations ≢ nothing
        (rgb, depth) = d.augmentations((rgb, depth))
    end

    if d.augmentations ≢ nothing
        rgb, depth = d.augmentations((rgb, depth))
    end
    rgb, depth = channelview.((rgb, depth))

    # return rgb, depth
    rgb = Float32.(permutedims(rgb, (3, 2, 1)))
    depth = Float32.(Flux.unsqueeze(permutedims(depth, (2, 1)), 3))

    rgb, depth
end

@inline DataLoaders.nobs(dataset::NYUDataset) = length(dataset)
@inline DataLoaders.getobs(dataset::NYUDataset, i) = dataset[i]

# d = NYUDataset("../nyu_data/data/nyu2_train/basement_0001a_out", (240,320))
