struct Model{E, D, P}
    encoder::E
    depth_decoder::D
    pose_decoder::P
end
Flux.@functor Model

function (m::Model)(x)
    features = m.encoder(x, Val(:stages))
    disparities = m.depth_decoder(features)
    disparities
end

eval_disparity(m::Model, x) = m.depth_decoder(m.encoder(x, Val(:stages)))
