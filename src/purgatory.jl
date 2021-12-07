#=
function full_obs_missmatch(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    p::Vector{Float64},
    q::Vector{Float64},
    vg::Vector{Float64},
    th_slack::Float64,
    thref::Vector{Float64},
    vref::Vector{Float64},
    pref::Vector{Float64},
    qref::Vector{Float64},
    mat::Matrices,
    id::Indices;
    Niter = 3::Int64,
    const_jac::Bool,
)
    b = -exp.(beta)
    g = exp.(gamma)
    th, v = newton_raphson_scheme(b, g, bsh, gsh, p, q, vg,
        th_slack, mat, id, Niter = Niter, const_jac = const_jac)
    p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat, id)

    return sum(abs.(th - thref)) + sum(abs.(v - vref)) +
        sum(abs.(p_est - pref)) + sum(abs.(q_est - qref)) 
end
=#


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



function batch_vth_based_train!(
    gm::GridModel,
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    opt;
    Niter::Int64 = 3,
    Ninter::Int64 = 10,
    Nepoch::Int64 = 10,
    const_jac::Bool = false,
    p_max::Float64 = 8.0,
)
    
    Nbatch = length(id_batch)
    ps = params(gm.beta, gm.gamma, gm.bsh, gm.gsh)
    for e = 1:Nepoch
        gs = gradient(ps) do
            b = -exp.(gm.beta)
            g = exp.(gm.gamma)
            th, v = newton_raphson_scheme(b, g, gm,bsh, gm.gsh,
                data.p[id.ns,1], data.q[id.pq,1], data.v[id.pv,1],
                data.th[id.slack, 1], mat, id, Niter = Niter,
                const_jac = const_jac)
            return sum(abs, th[id.ns] - data.th[id.ns,1]) +
                sum(abs, v[id.pv] - data.v[id.pv,1])
        end
        for i in 2:Nbatch
            gs .+= gradient(ps) do
                b = -exp.(gm.beta)
                g = exp.(gm.gamma)
                th, v = newton_raphson_scheme(b, g, gm.bsh, gm.gsh,
                    data.p[id.ns,i], data.q[is.pq,i], data.v[id.pv,i],
                    data.th[id.slack,i], mat, id, Niter = Niter,
                    const_jac = const_jac)
                return sum(abs, th[id.ns] - data.th[id.ns,i]) +
                    sum(abs, v[id.pv] - data.v[id.pv,i])
            end
        end
        Flux.update!(opt, ps, gs)
        #beta = min.(beta, p_max)
        #gamma = min.(gamma, p_max)
        if(mod(e, Ninter) == 0)
            error = 0
            error2 = 0
            for i in 1:Nbatch
                b = -exp.(gm.beta)
                g = exp.(gm.gamma)
                p_est, q_est = v2s_map(b, g, gm.bsh, gm.gsh, v, th, mat, id)
                error += (sum(abs.(th - thref[:,i])) + sum(abs.(v - vref[:,i]))) /
                    2.0 / prod(size(thref))
                error2 += (sum(abs.(p_est - pref[:,i])) + sum(abs.(q_est - qref[:,i]))) /
                    2.0 / prod(size(thref))
            end
            println([e, error, error2])
        end
    end
    return nothing
end


function batch_train!(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    p::Matrix{Float64},
    q::Matrix{Float64},
    vg::Matrix{Float64},
    th_slack::Vector{Float64},
    thref::Matrix{Float64},
    vref::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    mat::Matrices,
    id::Indices,
    opt;
    Niter::Int64 = 3,
    Ninter::Int64 = 10,
    Nepoch::Int64 = 10,
    const_jac::Bool = false,
    p_max::Float64 = 8.0,
)
    Nbatch = size(vg, 2)
    ps = params(beta, gamma, bsh, gsh)
    for e = 1:Nepoch
        gs = gradient(ps) do
            b = -exp.(beta)
            g = exp.(gamma)
            th, v = newton_raphson_scheme(b, g, bsh, gsh, p[:,1], q[:,1],
                vg[:,1], th_slack[1], mat, id, Niter = Niter,
                const_jac = const_jac)
            p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat, id)
            return sum(abs.(th - thref[:,1])) + sum(abs.(v - vref[:,1])) +
                sum(abs.(p_est - pref[:,1])) + sum(abs.(q_est - qref[:,1]))
        end
        for i in 2:Nbatch
            gs .+= gradient(ps) do
                b = -exp.(beta)
                g = exp.(gamma)
                th, v = newton_raphson_scheme(b, g, bsh, gsh, p[:,i], q[:,i],
                    vg[:,i], th_slack[i], mat, id, Niter = Niter,
                    const_jac = const_jac)
                p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat, id)
                return sum(abs.(th - thref[:,i])) + sum(abs.(v - vref[:,i])) +
                    sum(abs.(p_est - pref[:,i])) + sum(abs.(q_est - qref[:,i]))
            end
        end
        Flux.update!(opt, ps, gs)
        beta = min.(beta, p_max)
        gamma = min.(gamma, p_max)
        if(mod(e, Ninter) == 0)
            error = 0
            for i in 1:Nbatch
                b = -exp.(beta)
                g = exp.(gamma)
                th, v = newton_raphson_scheme(b, g, bsh, gsh, p[:,i], q[:,i], vg[:,i],
                    th_slack[i], mat, id, Niter = Niter, const_jac = const_jac)
                p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat, id)

                error += (sum(abs.(th - thref[:,i])) + sum(abs.(v - vref[:,i])) +
                    sum(abs.(p_est - pref[:,i])) + sum(abs.(q_est - qref[:,i]))) /
                    4.0 / prod(size(thref))
                
            end
            println([e, error])
        end
    end
    return nothing
end


#=
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
    mat::Matrices,
    id::Indices,
    opt;
    Ninter::Int64 = 5,
    Nepoch::Int64 = 10,
    beta_thres::Float64 = -3.0,
)
    Nbatch = size(v, 2)   
    id_kept = trues(size(beta))
    Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)
    ps = params(beta, gamma, bsh, gsh)
    for e in 1:Nepoch
        gs = gradient(ps) do
            b = -exp.(beta[id_kept])
            g = exp.(gamma[id_kept])
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            return sum(abs.(p-pref)) + sum(abs.(q-qref))
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            # if beta is smaller than a threshold, one ass
            id_kept = beta .> beta_thres
            mat = create_incidence_matrices(epsilon[id_kept,:], id)
            Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)
            b = -exp.(beta[id_kept])
            g = exp.(gamma[id_kept])
            p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
            error = (sum(abs.(p-pref)) + sum(abs.(q-qref)))  / 2.0 /
                prod(size(pref))
            println([e error sum(id_kept)])
            #ps = params(beta, gamma, bsh, gsh)
        end
    end
    #println(sum(id_kept))
    #println(beta[id_kept])
    #println(size(beta[id_kept]))
    #temp_beta = beta[id_kept]
    #temp_gamma = gamma[id_kept]
    #temp_epsilon = epsilon[id_kept]
    #global beta = temp_beta
    #global gamma = temp_gamma
    #global epsilon = temp_epsilon
    return id_kept
end
=#


#=
function V2S_loss(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    V2cos::Matrix{Float64},
    V2sin::Matrix{Float64},
    Vii::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    mat::Matrices,
)
    b = -exp.(beta)
    g = exp.(gamma)
    p, q = V2S_map(b, g, bsh, gsh, V2cos, V2sin, Vii, mat)
    return sum(abs.(p-pref)) + sum(abs.(q-qref))
end
=#


function batch_train!(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    p::Matrix{Float64},
    q::Matrix{Float64},
    vg::Matrix{Float64},
    th_slack::Vector{Float64},
    thref::Matrix{Float64},
    vref::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    mat::Matrices,
    id::Indices,
    opt;
    Niter = 3::Int64,
    Ninter = 10::Int64,
    Nepoch = 10::Int64,
    const_jac = false::Bool,
)
    Nbatch = size(vg, 2)
    for e = 1:Nepoch
        grad = (zeros(length(beta)), zeros(length(beta)),
            zeros(length(gsh)), zeros(length(bsh)))
        for i in 1:Nbatch
            g = gradient((beta, gamma, bsh, gsh) -> full_obs_missmatch(beta,
                gamma, bsh, gsh, p[:,i], q[:,i], vg[:,i],
                th_slack[i], thref[:,i], vref[:,i], pref[:,i],
                qref[:,i], mat, id, Niter = Niter, const_jac = const_jac),
                beta, gamma, bsh, gsh)
            grad[1] .+= g[1] / Nbatch
            grad[2] .+= g[2] / Nbatch
            grad[3] .+= g[3] / Nbatch
            grad[4] .+= g[4] / Nbatch
        end
        Flux.update!(opt, beta, grad[1])
        Flux.update!(opt, gamma, grad[2])
        Flux.update!(opt, bsh, grad[3])
        Flux.update!(opt, gsh, grad[4])
        if(mod(e, Ninter) == 0)
            error = 0
            for i in 1:Nbatch
                error += full_obs_missmatch(beta, gamma, bsh, gsh, p[:,i],
                q[:,i], vg[:,i], th_slack[i], thref[:,i], vref[:,i], pref[:,i],
                qref[:,i], mat, id, Niter = Niter, const_jac = const_jac)
            end
            println([e, error])
        end
    end
    return nothing
end
