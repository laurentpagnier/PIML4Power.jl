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
    
    # compute power injections (ie p and q)
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
    Ninter::Int64 = 5,
    Nepoch::Int64 = 10,
)
    Nbatch = size(id_batch, 2)    
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(data.th[:,id_batch], data.v[:,id_batch], mat, id)
    ps = params(parameters)
    println(ps)
    logs = zeros(0,3)
    for e in 1:Nepoch
        gs = gradient(ps) do
            b, g, bsh, gsh = param_fun(parameters)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            return sum(abs, p - data.p[:,id_batch]) + sum(abs, q - data.q[:, id_batch])
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            b, g, bsh, gsh = param_fun(parameters)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            loss = (sum(abs, p - data.p[:,id_batch]) + sum(abs, q - data.q[:,id_batch])) / 2.0 /
                Nbatch / id.Nbus
            dy = compare_params_2_admittance(b, g, bsh, gsh,
                data.b, data.g, data.bsh, data.gsh, mat)
            println([e loss dy])
            logs = vcat(logs, [e loss dy])
        end
    end  
    return logs
end


function train_n_update_V2S_map!(
    data::SystemData,
    epsilon::Matrix{Int64},
    id::Indices,
    opt,
    param_fun,
    parameters...;
    Ninter::Int64 = 5,
    Nepoch::Int64 = 10,
    beta_thres::Float64 = -3.0,
)
    Nbatch = size(id_batch, 2)
    for e in 1:Nepoch
        mat2 = create_incidence_matrices(epsilon_local, id)
        # learn the grid parameters
        train_V2S_map!(gm, data, mat, id, opt, param_fun,
            parameters, Ninter = 50, Nepoch = Ninter)
        # re-evaluate the structure of the grid
        id_kept = beta_local .> beta_thres
        epsilon_local = epsilon_local[id_kept,:]
        beta_local = beta_local[id_kept]
        gamma_local = gamma_local[id_kept]
        println([e, sum(id_kept)])
    end
    
    return beta_local, gamma_local, bsh, gsh, epsilon_local
end


function train_hybrid_V2S_map!(
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    nn::Flux.Chain,
    opt,
    param_fun,
    parameters...;
    Ninter::Int64 = 3,
    Nepoch::Int64 = 10,
    reg = 1
)
    Nbatch = size(id_batch, 2)
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(data.th[:,id_batch],
        data.v[:,id_batch], mat, id)
    ps = params(nn, parameters)
    nbias = prod(size(nn[end].bias))
    nw = prod(size(nn[end].weight))
    logs = zeros(0,5)
    
    for e in 1:Nepoch
        gs = gradient(ps) do
            b, g, bsh, gsh = param_fun(parameters)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            x = nn([data.v[:,id_batch]; data.th[:,id_batch]])
            ds = (sum(abs, p + x[1:id.Nbus,:] - data.p[:,id_batch]) +
                sum(abs, q + x[id.Nbus+1:end,:] - data.q[:,id_batch])) /
                2.0 / Nbatch / id.Nbus
            dw = (sum(abs, nn[end].weight) / nw +
                sum(abs, nn[end].bias) / nbias) 
            return ds  + reg * dw 
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            b, g, bsh, gsh = param_fun(parameters)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            x = nn([data.v[:,id_batch]; data.th[:,id_batch]])
            loss = (sum(abs, p + x[1:id.Nbus,:] - data.p[:,id_batch]) +
                sum(abs, q + x[id.Nbus+1:end,:] - data.q[:,id_batch])) / 2.0 /
                Nbatch / id.Nbus
            dy = compare_params_2_admittance(gm.beta, gm.gamma, gm.bsh, gm.gsh,
                data.b, data.g, data.bsh, data.gsh, mat)
            println([e loss maximum(abs.(x)) sum(abs.(x)) / prod(size(x)) dy])
            logs = vcat(logs, [e loss maximum(abs.(x)) sum(abs.(x)) / prod(size(x)) dy])
        end
    end  
    return logs
end
