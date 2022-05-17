export train_V2S_map!, train_n_update_V2S_map!, train_hybrid_V2S_map!

function preproc_V2S_map(
    th::Matrix{Float64},
    v::Matrix{Float64},
    mat::Matrices,
    id::Indices
)
    # some preprocessing for the V2S map that can be done once and for all
    dtheta = mat.Bmt * th
    Vij = (mat.Bint * v) .* (mat.Boutt * v)
    V2cos = Vij .* cos.(dtheta)
    V2sin = Vij .* sin.(dtheta)
    Vii = (v .* v)
    return Vij, V2cos, V2sin, Vii
end


function V2S_map(
    b::Vector{Float64},
    g::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    V2cos::Matrix{Float64},
    V2sin::Matrix{Float64},
    Vii::Matrix{Float64},
    mat::Matrices
)
    
    Gsh = mat.Bp * g + gsh
    Bsh = mat.Bp * b + bsh
    
    # compute active and reactive power injections (i.e. p and q)
    p = - mat.Bp * (g .* V2cos) + Gsh .* Vii -
        mat.Bm * (b .* V2sin)
    q = - mat.Bm * (g .* V2sin) - Bsh .* Vii +
        mat.Bp * (b .* V2cos)
    return p, q
end


function train_V2S_map!(
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    opt,
    param_fun,
    parameters...;
    Nepoch::Int64 = 100,
    Ninter::Int64 = 5,
)
    Nbatch = length(id_batch)  
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(data.th[:,id_batch],
        data.v[:,id_batch], mat, id)
    ps = params(parameters)
    logs = Dict{String,Any}("epochs" => Vector{Float64}([]),
        "loss" => Vector{Float64}([]),
        "dy" => Vector{Float64}([]))
    for e in 1:Nepoch
        gs = gradient(ps) do
            b, g, bsh, gsh = param_fun(parameters)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            return sum(abs2, p - data.p[:,id_batch]) +
                sum(abs2, q - data.q[:, id_batch]) / Nbatch / 
                id.Nbus
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            b, g, bsh, gsh = param_fun(parameters)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            loss = (sum(abs2, p - data.p[:,id_batch]) +
                sum(abs2, q - data.q[:,id_batch])) /
                Nbatch / id.Nbus
            dy = compare_params_2_admittance(b, g, bsh, gsh, id.epsilon, data.Y)
            println([e loss dy])
            append!(logs["epochs"], e)
            append!(logs["loss"], loss)
            append!(logs["dy"], dy)
        end
    end
    println(ps)
    return logs
end


function train_n_update_V2S_map!(
    data::SystemData,
    epsilon::Matrix{Int64},
    id::Indices,
    id_batch::Vector{Int64},
    opt,
    param_fun,
    red_param_fun, # see "params.jl" for more info on its role 
    parameters...;
    Nepoch::Int64 = 100,
    Ninter::Int64 = 5,
    tol::Float64 = 0.01,
    Nsubepoch::Int64 = 200,
    thres::Float64 = 0.05,
)
    param = parameters
    eps = epsilon
    id = create_indices(1, collect(1:id.Nbus), id.Nbus, epsilon)
    mat = create_incidence_matrices(id)
    for e in 1:Nepoch
        train_V2S_map!(data, mat, id, id_batch, opt, param_fun,
            param..., Ninter = Ninter, Nepoch = Nsubepoch)
            
        # re-evaluate the structure of the grid (i.e. if the susceptance
        # is smaller than a threshold, one assumes that there is no line)
        b, g, _, _ = param_fun(param)
        is_kept = sum(b^2 + g^2) .> thres^2
        
        # if there if any change, apply them
        if(sum(.!is_kept) > 0 & e != Nepoch)
            param = red_param_fun(param, is_kept)
            eps = eps[is_kept,:]
            id = create_indices(1, collect(1:id.Nbus), id.Nbus, eps)
            mat = create_incidence_matrices(id)
        end
        println([e, sum(is_kept), maximum(b), length(b)])
    end

    return eps, param...
end


function train_nn_hybrid_V2S_map!(
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    nn::Flux.Chain,
    opt,
    param_fun,
    parameters...;
    Nepoch::Int64 = 100,
    Ninter::Int64 = 5,
    lambda::Float64 = 0.1,
)
    Nbatch = length(id_batch)
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(data.th[:,id_batch],
        data.v[:,id_batch], mat, id)
    ps = params(nn, parameters)
    nbias = prod(size(nn[end].bias))
    nw = prod(size(nn[end].weight))
    logs = Dict{String,Any}("epochs" => Vector{Float64}([]),
        "loss" => Vector{Float64}([]),
        "dy" => Vector{Float64}([]),
        "max_nn_contrib" => Vector{Float64}([]),
        "avg_nn_contrib" => Vector{Float64}([]),)
    
    dv = data.v[:,id_batch] .- 1
    dth = data.th[:,id_batch] .- data.th[id.slack,1]
    
    for e in 1:Nepoch
        gs = gradient(ps) do
            b, g, bsh, gsh = param_fun(parameters)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            x = nn([dv; dth])
            ds = (sum(abs2, p + x[1:id.Nbus,:] - data.p[:,id_batch]) +
                sum(abs2, q + x[id.Nbus+1:end,:] - data.q[:,id_batch])) / 
                Nbatch / id.Nbus
            #dw = maximum(abs.(x[1:id.Nbus,:].^2 + x[id.Nbus+1:end,:].^2))
            dw = (sum(abs2, x[1:id.Nbus,:]) + sum(abs2, x[id.Nbus+1:end,:]))  / 
                Nbatch / id.Nbus
            return ds  + lambda * dw 
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            b, g, bsh, gsh = param_fun(parameters)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            x = nn([dv; dth])
            loss = (sum(abs2, p + x[1:id.Nbus,:] - data.p[:,id_batch]) +
                sum(abs2, q + x[id.Nbus+1:end,:] - data.q[:,id_batch])) /
                Nbatch / id.Nbus
            dy = compare_params_2_admittance(b, g, bsh, gsh, id.epsilon, data.Y)
            println([e loss maximum(abs.(x)) sum(abs.(x)) / prod(size(x)) dy])
            append!(logs["epochs"], e)
            append!(logs["loss"], loss)
            append!(logs["dy"], dy)
            append!(logs["max_nn_contrib"], maximum(abs.(x)))
            append!(logs["avg_nn_contrib"], sum(abs.(x)) / prod(size(x)))
        end
    end  
    return logs
end


function train_n_update_hybrid_V2S_map!(
    data::SystemData,
    epsilon::Matrix{Int64},
    id::Indices,
    id_batch::Vector{Int64},
    p_nn::Vector{Float64},
    q_nn::Vector{Float64},
    opt,
    param_fun,
    red_param_fun, # see "params.jl" for more info on its role 
    parameters...;
    Nepoch::Int64 = 100,
    Ninter::Int64 = 5,
    tol::Float64 = 0.01,
    Nsubepoch::Int64 = 200,
    lambda::Float64 = 0.1,
    thres::Float64 = 0.05,
)
    p0 = convert_param_tuple_into_vector(parameters)
    param = convert_param_vector_into_tuple(p0)
    eps = epsilon
    id = create_indices(1, collect(1:id.Nbus), id.Nbus, epsilon)
    mat = create_incidence_matrices(id)
    for e in 1:Nepoch
        train_vector_hybrid_V2S_map!(data, mat, id, id_batch, p_nn, q_nn,
            opt, param_fun, param..., Ninter = Ninter,
            Nepoch = Nsubepoch, lambda = lambda)
            
        # re-evaluate the structure of the grid (i.e. if the susceptance
        # is smaller than a threshold, one assumes that there is no line)
        b, g, _, _ = param_fun(param)
        is_kept = (b.^2 + g.^2) .> thres^2
        
        # if there if any change, apply them
        if(sum(.!is_kept) > 0 & e != Nepoch)
            red_param_fun(p0, is_kept)
            param = convert_param_vector_into_tuple(p0)
            println(param)
            eps = eps[is_kept,:]
            id = create_indices(1, collect(1:id.Nbus), id.Nbus, eps)
            mat = create_incidence_matrices(id)
        end
        println([e, sum(is_kept), maximum(b), length(b)])
    end

    return eps, param...
end


function train_vector_hybrid_V2S_map!(
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    p_nn::Vector{Float64},
    q_nn::Vector{Float64},
    opt,
    param_fun,
    parameters...;
    Nepoch::Int64 = 100,
    Ninter::Int64 = 5,
    lambda::Float64 = 0.1,
)
    Nbatch = length(id_batch)
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(data.th[:,id_batch],
        data.v[:,id_batch], mat, id)
    ps = params(p_nn, q_nn, parameters)
    logs = Dict{String,Any}("epochs" => Vector{Float64}([]),
        "loss" => Vector{Float64}([]),
        "dy" => Vector{Float64}([]),
        "max_nn_contrib" => Vector{Float64}([]),
        "avg_nn_contrib" => Vector{Float64}([]),)
    
    for e in 1:Nepoch
        gs = gradient(ps) do
            b, g, bsh, gsh = param_fun(parameters)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            ds = (sum(abs2, p .+ p_nn - data.p[:,id_batch]) +
                sum(abs2, q .+ q_nn - data.q[:,id_batch])) /
                Nbatch / id.Nbus   
            dw = (sum(abs2, p_nn) + sum(abs2, q_nn)) / id.Nbus
            return ds + lambda * dw
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            b, g, bsh, gsh = param_fun(parameters)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            loss = (sum(abs2, p .+ p_nn - data.p[:,id_batch]) +
                sum(abs2, q .+ q_nn - data.q[:,id_batch])) /
                Nbatch / id.Nbus 
            dy = compare_params_2_admittance(b, g, bsh, gsh, id.epsilon, data.Y)
            println([e loss maximum(abs, p_nn) sum(abs, p_nn) / prod(size(p_nn)) dy])
            append!(logs["epochs"], e)
            append!(logs["loss"], loss)
            append!(logs["dy"], dy)
            append!(logs["max_nn_contrib"], maximum(abs, p_nn))
            append!(logs["avg_nn_contrib"], sum(abs, p_nn) / prod(size(p_nn)))
        end
    end  
    return logs
end
