export train_V2S_map!, train_n_update_V2S_map!, train_hybrid_V2S_map!,
    load_hybrid_model, save_hybrid_model

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
    gm::GridModel,
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    opt;
    Ninter::Int64 = 5,
    Nepoch::Int64 = 10,
)
    Nbatch = size(id_batch, 2)    
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(data.th[:,id_batch], data.v[:,id_batch], mat, id)
    ps = params(gm.beta, gm.gamma, gm.bsh, gm.gsh)
    log = zeros(0,3)
    for e in 1:Nepoch
        gs = gradient(ps) do
            b = -exp.(gm.beta)
            g = exp.(gm.gamma)
            p, q = V2S_map(b, g, gm.bsh, gm.gsh, V2cos, V2sin, Vii, mat)
            return sum(abs, p - data.p[:,id_batch]) + sum(abs, q - data.q[:, id_batch])
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            b = -exp.(gm.beta)
            g = exp.(gm.gamma)
            p, q = V2S_map(b, g, gm.bsh, gm.gsh, V2cos, V2sin, Vii, mat)
            loss = (sum(abs, p - data.p[:,id_batch]) + sum(abs, q - data.q[:,id_batch])) / 2.0 /
                Nbatch / id.Nbus
            dy = compare_params_2_admittance(gm.beta, gm.gamma, gm.bsh, gm.gsh,
                data.b, data.g, data.bsh, data.gsh, mat)
            println([e loss dy])
            log = vcat(log, [e loss dy])
        end
    end  
    return log
end


function train_n_update_V2S_map!(
    gm::GridModel,
    data::SystemData,
    epsilon::Matrix{Int64},
    id::Indices,
    opt;
    Ninter::Int64 = 5,
    Nepoch::Int64 = 10,
    beta_thres::Float64 = -3.0,
)
    Nbatch = size(id_batch, 2)
    for e in 1:Nepoch
        mat2 = create_incidence_matrices(epsilon_local, id)
        # if beta is smaller than a threshold, one ass
        train_V2S_map!(gm, data, mat,
            id, opt, Ninter = 50, Nepoch = Ninter)
        id_kept = beta_local .> beta_thres
        epsilon_local = epsilon_local[id_kept,:]
        beta_local = beta_local[id_kept]
        gamma_local = gamma_local[id_kept]
        println([e, sum(id_kept)])
    end
    
    return beta_local, gamma_local, bsh, gsh, epsilon_local
end


function train_hybrid_V2S_map!(
    gm::GridModel,
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    nn::Flux.Chain,
    opt;
    Ninter::Int64 = 3,
    Nepoch::Int64 = 10,
    reg = 1
)
    Nbatch = size(id_batch, 2)
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(data.th[:,id_batch],
        data.v[:,id_batch], mat, id)
    ps = params(nn, gm.beta, gm.gamma, gm.bsh, gm.gsh)
    nbias = prod(size(nn[end].bias))
    nw = prod(size(nn[end].weight))
    log = zeros(0,5)
    
    for e in 1:Nepoch
        gs = gradient(ps) do
            b = -exp.(gm.beta)
            g = exp.(gm.gamma)
            p, q = V2S_map(b, g, gm.bsh, gm.gsh, V2cos, V2sin, Vii, mat)
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
            b = -exp.(gm.beta)
            g = exp.(gm.gamma)
            p, q = V2S_map(b, g, gm.bsh, gm.gsh, V2cos, V2sin, Vii, mat)
            x = nn([data.v[:,id_batch]; data.th[:,id_batch]])
            loss = (sum(abs, p + x[1:id.Nbus,:] - data.p[:,id_batch]) +
                sum(abs, q + x[id.Nbus+1:end,:] - data.q[:,id_batch])) / 2.0 /
                Nbatch / id.Nbus
            dy = compare_params_2_admittance(gm.beta, gm.gamma, gm.bsh, gm.gsh,
                data.b, data.g, data.bsh, data.gsh, mat)
            println([e loss maximum(abs.(x)) sum(abs.(x)) / prod(size(x)) dy])
            log = vcat(log, [e loss maximum(abs.(x)) sum(abs.(x)) / prod(size(x)) dy])
        end
    end  
    return log
end


function test_hybrid_V2S_map(
    gm::GridModel,
    data::SystemData,
    mat::Matrices,
    id::Indices,
    nn::Flux.Chain,
)
    b = -exp.(gm.beta)
    g = exp.(gm.gamma)
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(data.th, data.v, mat, id)
    p, q = V2S_map(b, g, gm.bsh, gm.gsh, V2cos, V2sin, Vii, mat)
    x = nn([v;th])
    error = (sum(abs, p + x[1:id.Nbus,:] - data.p) +
    sum(abs, q + x[id.Nbus+1:end,:] - data.q)) / 2.0 /
    prod(size(pref))
    return error, maximum(abs.(x)), sum(abs.(x)) / prod(size(x))
end


function test_hybrid_V2S_map(
    gm::GridModel,
    data::SystemData,
    yref::Vector{ComplexF64},
    ysh_ref::Vector{ComplexF64},
    mat::Matrices,
    id::Indices,
    nn::Flux.Chain,
)
    error, max_x, avg_x = test_hybrid_V2S_map(gm, data, mat, id, nn)
    dy = compare_params_2_admittance(beta, gamma, bsh, gsh, imag(yref),
        real(yref), imag(ysh_ref), real(ysh_ref), mat)
    return error, max_x, avg_x, dy
end


function save_grid_model(
    gm::GridModel,
    rootname::String,
)
    HDF5.h5open(rootname * ".h5","w") do fid
        fid["/beta"] = gm.beta
        fid["/gamma"] = gm.gamma
        fid["/bsh"] = gm.bsh
        fid["/gsh"] = gm.gsh
        fid["/epsilon"] = gm.epsilon
        close(fid)
    end
    return nothing
end


function save_hybrid_model(
    gm::GridModel,
    nn, # a Flux neural network (i.e. Chain)
    rootname::String,
)
    save_grid_model(gm, rootname)  
    @save rootname * ".bson" nn
    return nothing
end


function load_grid_model(
    rootname::String,
)
    param = h5read(rootname * ".h5","/")
    beta = param["beta"]
    gamma = param["gamma"]
    bsh = param["bsh"]
    gsh = param["gsh"]
    epsilon = Int64.(param["epsilon"])
    id = create_indices(1, collect(1:maximum(epsilon)),
        maximum(epsilon), epsilon) # dummy way to do so as we mostly want epsilon
    mat = create_incidence_matrices(id)
    return create_gridmodel(id.epsilon, beta, gamma, bsh, gsh)
end


function load_hybrid_model(
    rootname::String,
)
    beta, gamma, bsh, gsh, mat, id = load_grid_model(rootname)
    @load rootname * ".bson" nn
    return beta, gamma, bsh, gsh, mat, id, nn
end


function create_simple_full_nn(
    N::Int64;
    act_fun = tanh
)
    return c = Chain(
        Dense(2*N, 2*N, act_fun),
        Dense(2*N, 2*N, act_fun),
        Dense(2*N, 2*N, act_fun),
        Dense(2*N, 2*N, act_fun),
        Dense(2*N, 2*N)
        )
end
