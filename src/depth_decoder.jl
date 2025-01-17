struct DecoderBlock{D}
    d::D
end
Flux.@functor DecoderBlock
(block::DecoderBlock)(x) = block.d(pad_reflect(x, 1))

struct BranchBlock{C1, C2}
    c1::C1
    c2::C2
end
Flux.@functor BranchBlock
function BranchBlock(in_channels, skip_channels, out_channels)
    c1 = DecoderBlock(Conv((3, 3), in_channels=>out_channels, elu))
    c2 = DecoderBlock(Conv((3, 3), (out_channels + skip_channels)=>out_channels, elu))
    BranchBlock(c1, c2)
end

(b::BranchBlock)(x, ::Nothing) = b.c2(upsample_bilinear(b.c1(x), (2, 2)))
(b::BranchBlock)(x, skip) = b.c2(cat(upsample_bilinear(b.c1(x), (2, 2)), skip; dims=3))

struct DepthDecoder{B, D}
    branches::B
    decoders::D
end
Flux.@functor DepthDecoder
function DepthDecoder(;encoder_channels, scale_levels, embedding_levels=21)
    if length(scale_levels) > 5 || minimum(scale_levels) < 1 || maximum(scale_levels) > 5
        error("`scale_levels` should be at most of length 5 and have values in [1, 5] range.")
    end

    decoder_channels = [256, 128, 64, 32, 16]
    encoder_channels = encoder_channels[end:-1:1] .+ embedding_levels
    head_channels = encoder_channels[1]
    in_channels = [head_channels, decoder_channels[1:end - 1]...]
    skip_channels = [encoder_channels[2:end]..., 0]

    bstart = 1
    branches, decoders = [], []
    for slevel in scale_levels
        push!(branches, [
            BranchBlock(in_channels[bid], skip_channels[bid], decoder_channels[bid])
            for bid in bstart:slevel])
        # TODO: set correct number of output channels (color+density) 
        # TODO: sigmoid activation for output?

        push!(decoders, DecoderBlock(Conv((3, 3), decoder_channels[slevel]=>1, σ)))
        bstart = slevel + 1
    end
    DepthDecoder(branches, decoders)
end

function (d::DepthDecoder)(features::NTuple{N, T})::Vector{T} where {T, N}
    x, skips = features[end], features[(end - 1):-1:1]
    n_skips = length(skips)
    bstart = 1

    function _runner(branch, decoder)
        bend = bstart + length(branch) - 1
        brange = bstart:bend
        bstart = bend + 1

        x = foldl((x, 1:length(branch)...)) do o, i
            branch[i](o, brange[i] ≤ n_skips ? skips[brange[i]] : nothing)
        end
        decoder(x)
    end
    map(k -> _runner(k...), zip(d.branches, d.decoders))
end
