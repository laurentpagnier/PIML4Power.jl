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
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    th::Matrix{Float64},
    v::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    mat::Matrices,
    id::Indices,
    opt;
    Ninter::Int64 = 5,
    Nepoch::Int64 = 10,
)
    Nbatch = size(v,2)    
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)
    ps = params(beta, gamma, bsh, gsh)
    for e in 1:Nepoch
        gs = gradient(ps) do
            b = -exp.(beta)
            g = exp.(gamma)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            return sum(abs.(p-pref)) + sum(abs.(q-qref))
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            b = -exp.(beta)
            g = exp.(gamma)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            error = (sum(abs.(p-pref)) + sum(abs.(q-qref)))  / 2.0 /
                prod(size(pref))
            println([e error])
        end
    end  
    return nothing
end


function train_n_update_V2S_map!(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    th::Matrix{Float64},
    v::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    epsilon::Matrix{Int64},
    id::Indices,
    opt;
    Ninter::Int64 = 5,
    Nepoch::Int64 = 10,
    beta_thres::Float64 = -3.0,
)
    Nbatch = size(v, 2)
    beta_local = copy(beta)
    gamma_local = copy(gamma)
    epsilon_local = copy(epsilon)
    for e in 1:Nepoch
        mat2 = create_incidence_matrices(epsilon_local, id)
        # if beta is smaller than a threshold, one ass
        train_V2S_map!(beta_local, gamma_local, bsh, gsh, th, v, pref, qref, mat2,
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
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    th::Matrix{Float64},
    v::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    mat::Matrices,
    id::Indices,
    c::Flux.Chain,
    opt;
    Ninter::Int64 = 3,
    Nepoch::Int64 = 10,
    reg = 1
)
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)
    ps = params(c, beta, gamma, bsh, gsh)
    nbias = prod(size(c[end].bias))
    nw = prod(size(c[end].weight))
    for e in 1:Nepoch
        gs = gradient(ps) do
            b = -exp.(beta)
            g = exp.(gamma)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            x = c([v;th])
            ds = (sum(abs, p + x[1:id.Nbus,:] - pref) +
                sum(abs, q + x[id.Nbus+1:end,:] - qref)) /
                2.0 / prod(size(pref))
            dw = (sum(abs, c[end].weight) / nw +
                sum(abs, c[end].bias) / nbias) 
            return ds  + reg * dw 
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            b = -exp.(beta)
            g = exp.(gamma)
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            x = c([v;th])
            error = (sum(abs, p + x[1:id.Nbus,:] - pref) +
                sum(abs, q + x[id.Nbus+1:end,:] - qref)) / 2.0 /
                prod(size(pref))
            println([e error maximum(abs.(x)) sum(abs.(x)) / prod(size(x))])
        end
    end  
    return nothing
end


function test_hybrid_V2S_map(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    th::Matrix{Float64},
    v::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    mat::Matrices,
    id::Indices,
    nn::Flux.Chain,
)
    b = -exp.(beta)
    g = exp.(gamma)
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)
    p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
    x = nn([v;th])
    error = (sum(abs, p + x[1:id.Nbus,:] - pref) +
    sum(abs, q + x[id.Nbus+1:end,:] - qref)) / 2.0 /
    prod(size(pref))
    return error, maximum(abs.(x)), sum(abs.(x)) / prod(size(x))
end


function test_hybrid_V2S_map(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    th::Matrix{Float64},
    v::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    yref::Vector{ComplexF64},
    ysh_ref::Vector{ComplexF64},
    mat::Matrices,
    id::Indices,
    nn::Flux.Chain,
)
    error, max_x, avg_x = test_hybrid_V2S_map(beta, gamma, bsh, gsh, th, v, pref, qref,
        mat, id, nn)
    dy = compare_params_2_admittance(beta, gamma, bsh, gsh, imag(yref),
        real(yref), imag(ysh_ref), real(ysh_ref), mat)
    return error, max_x, avg_x, dy
end


function save_hybrid_model(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    epsilon::Matrix{Int64},
    nn, # a Flux neural network (i.e. Chain)
    rootname::String,
)
    if(isfile(rootname * ".h5"))
        HDF5.h5open(rootname * ".h5","w") do fid
            fid["/beta"] = beta
            fid["/gamma"] = gamma
            fid["/bsh"] = bsh
            fid["/gsh"] = gsh
            fid["/epsilon"] = epsilon
            close(fid)
        end
    else
        h5write(rootname * ".h5", "/beta", beta)
        h5write(rootname * ".h5", "/gamma", gamma)
        h5write(rootname * ".h5", "/bsh", bsh)
        h5write(rootname * ".h5", "/gsh", gsh)
        h5write(rootname * ".h5", "/epsilon", epsilon)
    end
    @save rootname * ".bson" nn
    return nothing
end


function load_hybrid_model(
    rootname::String,
)
    param = h5read(rootname * ".h5","/")
    beta = param["beta"]
    gamma = param["gamma"]
    bsh = param["bsh"]
    gsh = param["gsh"]
    epsilon = Int64.(param["epsilon"])
    @load rootname * ".bson" nn
    id = create_indices(1, collect(1:maximum(epsilon)),
        maximum(epsilon), epsilon) # dummy way to do so as we mostly want epsilon
    mat = create_incidence_matrices(id)
    return beta, gamma, bsh, gsh, mat, id, nn
end


function create_simple_full_nn(
    N::Int64;
    act_fun = tanh
)
    return c = Chain(Dense(2*N,2*Ntanh,act_fun), Dense(2*N,2*N, act_fun),
        Dense(2*N,2*N,act_fun), Dense(2*N,2*N))
end
