export newton_raphson_scheme, v2s_map, batch_train!

using Zygote

function batch_train_with_pmus!(
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    opt,
    param_fun,
    parameters...;
    Niter::Int64 = 3,
    Ninter::Int64 = 10,
    Nepoch::Int64 = 10,
    const_jac::Bool = false,
    p_max::Float64 = 8.0,
)
    Nbatch = length(id_batch)
    ps = params(parameters)
    
    # here we will also treat PV buses as PQ buses
    id_temp = create_indices(id.slack, [id.slack], id.Nbus, id.epsilon)    
    #println(id_temp.pq)
    mat_temp = create_incidence_matrices(id_temp)
    logs = Dict{String,Any}("epochs" => Vector{Float64}([]),
        "loss" => Vector{Float64}([]),
        "dy" => Vector{Float64}([]))
    for e = 1:Nepoch
        
        grads = IdDict()
        grads[ps[2]] = zeros(size(ps[2]))
        grads[ps[4]] = zeros(size(ps[4]))
        grads[ps[1]] = zeros(size(ps[1]))
        grads[ps[3]] = zeros(size(ps[3]))
        gs = Zygote.Grads(grads, ps)
        Threads.@threads for i in id_batch
            gs .+= gradient(ps) do
                b, g, bsh, gsh = param_fun(parameters)
                th, v = newton_raphson_scheme(b, g, bsh, gsh,
                    data.p[id_temp.ns,i], data.q[id_temp.pq,i], data.v[id_temp.pv,i],
                    data.th[id_temp.slack,i], mat_temp, id_temp, Niter = Niter,
                    const_jac = const_jac)
                    
                p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat_temp, id_temp)
                
                return sum(abs, v[id.pv] - data.v[id.pv,i]) / length(id.pv) / 0.1 +
                    sum(abs, th[id.pv] - data.th[id.pv,i]) / length(id.pv) / 0.52 + # 0.52 approx 30deg
                    sum(abs, q_est - data.q[:,i]) / id.Nbus / maximum(data.p) +
                    sum(abs, p_est - data.p[:,i]) / id.Nbus / maximum(data.p) 
            end
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            loss_th = 0
            loss_v = 0
            loss_p = 0
            loss_q = 0
            b, g, bsh, gsh = param_fun(parameters)
            Threads.@threads for i in id_batch
                th, v = newton_raphson_scheme(b, g, bsh, gsh, data.p[id.ns,i],
                    data.q[id.pq,i], data.v[id.pv,i], data.th[id.slack,i],
                    mat, id, Niter = Niter, const_jac = const_jac)
                p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat, id)
                
                loss_v += sum(abs, th - data.th[:,i])
                loss_th += sum(abs, th - data.th[:,i])
                loss_p += sum(abs, p_est - data.p[:,i])
                loss_q += sum(abs, q_est - data.q[:,i])

            end
            dy = compare_params_2_admittance(b, g, bsh, gsh,
                data.b, data.g, data.bsh, data.gsh, mat)
            println([e, loss_th, loss_v, loss_p, loss_q, dy])
            append!(logs["epochs"], e)
            append!(logs["loss"], loss_th + loss_p + loss_q)
            append!(logs["dy"], dy)
        end
    end
    return logs
end


function batch_train_pq!(
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    opt,
    param_fun,
    parameters...;
    Niter::Int64 = 3,
    Ninter::Int64 = 10,
    Nepoch::Int64 = 10,
    const_jac::Bool = false,
    p_max::Float64 = 8.0,
)
    Nbatch = length(id_batch)
    ps = params(parameters)

    logs = Dict{String,Any}("epochs" => Vector{Float64}([]),
        "loss" => Vector{Float64}([]),
        "dy" => Vector{Float64}([]))
    for e = 1:Nepoch
        
        grads = IdDict()
        grads[ps[1]] = zeros(size(ps[1]))
        grads[ps[2]] = zeros(size(ps[2]))
        grads[ps[3]] = zeros(size(ps[3]))
        grads[ps[4]] = zeros(size(ps[4]))
        gs = Zygote.Grads(grads, ps)
        Threads.@threads for i in id_batch
            gs .+= gradient(ps) do
                b, g, bsh, gsh = param_fun(parameters)
                th, v = newton_raphson_scheme(b, g, bsh, gsh,
                    data.p[id.ns,i], data.q[id.pq,i], data.v[id.pv,i],
                    data.th[id.slack,i], mat, id, Niter = Niter,
                    const_jac = const_jac)
                    
                p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat, id)
                    
                return sum(abs, p_est - data.p[:,i]) +
                    sum(abs, q_est - data.q[:,i]) 
            end
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            loss_th = 0
            loss_v = 0
            loss_p = 0
            loss_q = 0
            b, g, bsh, gsh = param_fun(parameters)
            Threads.@threads for i in id_batch
                th, v = newton_raphson_scheme(b, g, bsh, gsh, data.p[id.ns,i],
                    data.q[id.pq,i], data.v[id.pv,i], data.th[id.slack,i],
                    mat, id, Niter = Niter, const_jac = const_jac)
                p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat, id)
                loss_th += sum(abs, th - data.th[:,i])
                loss_v += sum(abs, v - data.v[:,i])
                loss_p += sum(abs, p_est - data.p[:,i])
                loss_q += sum(abs, q_est - data.q[:,i])
            end
            dy = compare_params_2_admittance(b, g, bsh, gsh,
                data.b, data.g, data.bsh, data.gsh, mat)
            println([e, loss_th, loss_v, loss_p, loss_q, dy])
            append!(logs["epochs"], e)
            append!(logs["loss"], loss_th + loss_p + loss_q)
            append!(logs["dy"], dy)
        end
    end
    return logs
end


function batch_train_vth!(
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    opt,
    param_fun,
    parameters...;
    Niter::Int64 = 3,
    Ninter::Int64 = 10,
    Nepoch::Int64 = 10,
    const_jac::Bool = false,
    p_max::Float64 = 8.0,
)
    # in this configuration, every bus is assumed to be PQ (expect the slack bus)
    id_temp = Indices(id.slack, [id.slack], id.ns, id.ns, id.Nbus, id.epsilon)
    mat_temp = create_incidence_matrices(id_temp)
    
    Nbatch = length(id_batch)
    ps = params(parameters)

    logs = Dict{String,Any}("epochs" => Vector{Float64}([]),
        "loss" => Vector{Float64}([]),
        "dy" => Vector{Float64}([]))
    for e = 1:Nepoch
        
        grads = IdDict()
        grads[ps[2]] = zeros(size(ps[2]))
        grads[ps[4]] = zeros(size(ps[4]))
        grads[ps[1]] = zeros(size(ps[1]))
        grads[ps[3]] = zeros(size(ps[3]))
        gs = Zygote.Grads(grads, ps)
        Threads.@threads for i in id_batch
            gs .+= gradient(ps) do
                b, g, bsh, gsh = param_fun(parameters)
                th, v = newton_raphson_scheme(b, g, bsh, gsh,
                    data.p[id_temp.ns,i], data.q[id_temp.pq,i], data.v[id_temp.pv,i],
                    data.th[id_temp.slack,i], mat_temp, id_temp, Niter = Niter,
                    const_jac = const_jac)
                    
                return sum(abs2, th[id_temp.ns] - data.th[id_temp.ns,i]) / 0.52 / length(id_temp.ns) +
                    sum(abs2, v[id_temp.ns] - data.v[id_temp.ns,i]) / 0.1 / length(id_temp.ns) 
            end
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            loss_th = 0
            loss_v = 0
            loss_p = 0
            loss_q = 0
            b, g, bsh, gsh = param_fun(parameters)
            Threads.@threads for i in id_batch
                th, v = newton_raphson_scheme(b, g, bsh, gsh, data.p[id.ns,i],
                    data.q[id.pq,i], data.v[id.pv,i], data.th[id.slack,i],
                    mat, id, Niter = Niter, const_jac = const_jac)
                p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat, id)
                loss_th += sum(abs, th - data.th[:,i])
                loss_v += sum(abs, v - data.v[:,i])
                loss_p += sum(abs, p_est - data.p[:,i])
                loss_q += sum(abs, q_est - data.q[:,i])
            end
            dy = compare_params_2_admittance(b, g, bsh, gsh,
                data.b, data.g, data.bsh, data.gsh, mat)
            println([e, loss_th, loss_v, loss_p, loss_q, dy])
            append!(logs["epochs"], e)
            append!(logs["loss"], loss_th + loss_p + loss_q)
            append!(logs["dy"], dy)
        end
    end
    return logs
end


function batch_train!(
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64},
    opt,
    param_fun,
    parameters...;
    Niter::Int64 = 3,
    Ninter::Int64 = 10,
    Nepoch::Int64 = 10,
    const_jac::Bool = false,
    p_max::Float64 = 8.0,
)
    Nbatch = length(id_batch)
    ps = params(parameters)

    logs = Dict{String,Any}("epochs" => Vector{Float64}([]),
        "loss" => Vector{Float64}([]),
        "dy" => Vector{Float64}([]))
    for e = 1:Nepoch
        
        grads = IdDict()
        grads[ps[2]] = zeros(size(ps[2]))
        grads[ps[4]] = zeros(size(ps[4]))
        grads[ps[1]] = zeros(size(ps[1]))
        grads[ps[3]] = zeros(size(ps[3]))
        gs = Zygote.Grads(grads, ps)
        Threads.@threads for i in id_batch
            gs .+= gradient(ps) do
                b, g, bsh, gsh = param_fun(parameters)
                th, v = newton_raphson_scheme(b, g, bsh, gsh,
                    data.p[id.ns,i], data.q[id.pq,i], data.v[id.pv,i],
                    data.th[id.slack,i], mat, id, Niter = Niter,
                    const_jac = const_jac)
                    
                p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat, id)
                #=
                return sum(abs, p_est - data.p[:,i]) +
                    sum(abs, q_est - data.q[:,i]) +
                    sum(abs, th[id.pv] - data.th[id.pv,i])
                =#
                    
                return sum(abs, th[id.pv] - data.th[id.pv,i]) / length(id.pv) / 0.52 +
                    sum(abs, v[id.pv] - data.v[id.pv,i]) / length(id.pv) / 0.1 +
                    sum(abs, p_est - data.p[:,i]) / id.Nbus / maximum(data.p) +
                    sum(abs, q_est - data.q[:,i]) / id.Nbus / maximum(data.p) 
            end
        end
        
        Flux.update!(opt, ps, gs)
        
        if(mod(e, Ninter) == 0)
            loss_th = 0
            loss_v = 0
            loss_p = 0
            loss_q = 0
            b, g, bsh, gsh = param_fun(parameters)
            Threads.@threads for i in id_batch
                th, v = newton_raphson_scheme(b, g, bsh, gsh, data.p[id.ns,i],
                    data.q[id.pq,i], data.v[id.pv,i], data.th[id.slack,i],
                    mat, id, Niter = Niter, const_jac = const_jac)
                p_est, q_est = v2s_map(b, g, bsh, gsh, v, th, mat, id)
                loss_th += sum(abs, th - data.th[:,i])
                loss_v += sum(abs, v - data.v[:,i])
                loss_p += sum(abs, p_est - data.p[:,i])
                loss_q += sum(abs, q_est - data.q[:,i])
            end
            dy = compare_params_2_admittance(b, g, bsh, gsh,
                data.b, data.g, data.bsh, data.gsh, mat)
            println([e, loss_th, loss_v, loss_p, loss_q, dy])
            append!(logs["epochs"], e)
            append!(logs["loss"], loss_th + loss_p + loss_q)
            append!(logs["dy"], dy)
        end
    end
    return logs
end


function newton_raphson_scheme(
    b::Vector{Float64},
    g::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    p::Vector{Float64},
    q::Vector{Float64},
    vg::Vector{Float64},
    th_slack::Float64,
    mat::Matrices,
    id::Indices;
    Niter::Int64 = 3,
    const_jac::Bool = false,
)
    v = zeros(id.Nbus)
    th = zeros(id.Nbus)
    v = v + mat.pv2full * vg  + mat.pq2full * ones(length(id.pq)) 
    th = th + vec(mat.s2full * th_slack)

    v_new = zeros(id.Nbus)
    th_new = zeros(id.Nbus)

    if const_jac
        jac = jacob(b, g, bsh, gsh, v, th, mat, id)
    end

    for i in 1:Niter
        dpq = dPQ(b, g, bsh, gsh, v, th, mat, id, p, q)
        if const_jac == false
            jac = jacob(b, g, bsh, gsh, v, th, mat, id)
        end
        x = jac \ dpq
        #x = idrs(jac, dpq,s=1) # blow the memory during the compiling!
        th_new = th - mat.ns2full * x[1:id.Nbus-1]
        v_new = v - mat.pq2full * x[id.Nbus:end]
        v = v_new
        th = th_new
    end
    
    return th, v
end


function dPQ(
    b::Vector{Float64},
    g::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    v::Vector{Float64},
    th::Vector{Float64},
    mat::Matrices,
    id::Indices,
    p_ref::Vector{Float64},
    q_ref::Vector{Float64}
)
    # compute the missmatches for NR scheme
    dtheta = mat.Bmt * vec(th)
    costh = cos.(dtheta) 
    sinth = sin.(dtheta) 

    gs = g .* sinth
    gc = g .* costh
    bs = b .* sinth
    bc = b .* costh

    p = v[id.ns] .* ( (mat.Bin_ns * ((-gc - bs) .* mat.Boutt) +
        mat.Bout_ns * ((-gc + bs) .* mat.Bint)) * v +
        (mat.Bp_ns * g + gsh[id.ns]) .* v[id.ns])
    
    q = v[id.pq] .* ( (mat.Bin_pq * ((-gs + bc) .* mat.Boutt) +
        mat.Bout_pq * ((gs + bc) .* mat.Bint)) * v -
        (mat.Bp_pq * b + bsh[id.pq]) .* v[id.pq])

    dp = p - p_ref
    dq = q - q_ref
    return [dp; dq]
end


function jacob(
    b::Vector{Float64},
    g::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    v::Vector{Float64},
    th::Vector{Float64},
    mat::Matrices,
    id::Indices
)
    # compute the jacobian matrix for NR scheme
    
    dtheta = mat.Bmt * th
    dtheta_ns = mat.Bm_nst * th[id.ns]

    costh = cos.(dtheta)
    sinth = sin.(dtheta)
    
    gs = g .* sinth
    gc = g .* costh
    bs = b .* sinth
    bc = b .* costh

    jac1 = v[id.ns] .* (
            mat.Bin_ns * ((-gs + bc) .* mat.Bout_nst) +
            mat.Bout_ns * ((gs + bc) .* mat.Bin_nst)
        ) .* reshape(v[id.ns], 1, length(id.ns))
    j1 = (v[id.ns] .* (mat.Bin_ns * ((gs - bc) .* mat.Boutt) +
        mat.Bout_ns * ((-gs - bc) .* mat.Bint))) * v
    jac1b = jac1 + mat.Ins .* j1

    jac2 = v[id.ns] .* (
            mat.Bin_ns * ((-gc - bs) .* mat.Bout_pqt) +
            mat.Bout_ns * ((-gc + bs) .* mat.Bin_pqt)
        )
    j2 = (mat.Bin_pq * ((-gc - bs) .* mat.Boutt) +
        mat.Bout_pq * ((-gc + bs) .* mat.Bint) ) * v
    j2b = j2 + 2 * diagm(mat.Bp_pq * g + gsh[id.pq]) * v[id.pq]  
    jac2b = jac2 + mat.pq2ns .* reshape(j2b, 1, length(j2b))
 
    jac3 = v[id.pq] .* (
            mat.Bin_pq * ((gc + bs) .* mat.Bout_nst) +
            mat.Bout_pq * ((gc - bs) .* mat.Bin_nst)
        ) .* reshape(v[id.ns], 1, length(id.ns))
    j3 = (v[id.pq] .* (mat.Bin_pq * ((-gc - bs) .* mat.Boutt) +
        mat.Bout_pq * ((-gc + bs) .* mat.Bint))) * v
    jac3b = jac3 + j3 .* mat.pq2nst  
    
    jac4 = v[id.pq] .* (
            mat.Bin_pq * ((-gs + bc) .* mat.Bout_pqt) +
            mat.Bout_pq * ((gs + bc) .* mat.Bin_pqt)
        )
    j4 = (mat.Bin_pq * ((-gs + bc) .* mat.Boutt) +
        mat.Bout_pq * ((gs + bc) .* mat.Bint)) * v
    j4b = j4 - 2 * diagm(mat.Bp_pq * b + bsh[id.pq]) * v[id.pq]
    jac4b = jac4 + mat.Ipq .* j4b
    
    return [jac1b jac2b; jac3b jac4b]
    
end


function v2s_map(
    b::Vector{Float64},
    g::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    v::Vector{Float64},
    th::Vector{Float64},
    mat::Matrices,
    id::Indices
)
    # compute the missmatches for NR scheme
    dtheta = mat.Bmt * vec(th)
    costh = cos.(dtheta) 
    sinth = sin.(dtheta) 

    gs = g .* sinth
    gc = g .* costh
    bs = b .* sinth
    bc = b .* costh

    p = v .* ( (mat.Bin * ((-gc - bs) .* mat.Boutt) +
        mat.Bout * ((-gc + bs) .* mat.Bint)) * v  +
        (mat.Bp * g + gsh) .* v)
    
    q = v .* ( (mat.Bin * ((-gs + bc) .* mat.Boutt) +
        mat.Bout * ((gs + bc) .* mat.Bint)) * v -
        (mat.Bp * b + bsh) .* v)
    return p, q
end




