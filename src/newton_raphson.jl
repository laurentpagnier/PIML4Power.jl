export newton_raphson_scheme, v2s_map, batch_train!

using Zygote

function batch_with_v2p_train!(
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
    Nstep::Int64 = 100,
)
    Nbatch = length(id_batch)
    ps = params(gm.beta, gm.gamma, gm.bsh, gm.gsh)
    for e = 1:Nepoch
    
        v = zeros(id.Nbus, Nbatch)
        th = zeros(id.Nbus, Nbatch)
        for i in 1:Nbatch
            b = -exp.(gm.beta)
            g = exp.(gm.gamma)
                
            th_i, v_i = newton_raphson_scheme(b, g, gm.bsh, gm.gsh,
                data.p[id.ns,i], data.q[id.pq,i], data.v[id.pv,i],
                data.th[id.slack,i], mat, id, Niter = Niter,
                const_jac = const_jac)
            
            th[:,i] = th_i
            v[:,i] = v_i
        end
        
        # V2S map
        Vij, V2cos, V2sin, Vii = preproc_V2S_map(th, v, mat, id)
        for _ in 1:Nstep
            gs = gradient(ps) do
                b = -exp.(gm.beta)
                g = exp.(gm.gamma)
                p, q = V2S_map(b, g, gm.bsh, gm.gsh, V2cos, V2sin, Vii, mat)
                return sum(abs, p[id.slack,:] - data.p[id.slack,id_batch]) +
                    sum(abs, q[id.pv, :] - data.q[id.pv, id_batch])
            end
            Flux.update!(opt, ps, gs)
        end 
        
        if(mod(e, Ninter) == 0)
            loss_vth = 0
            loss_pq = 0
            for i in 1:Nbatch
                b = -exp.(gm.beta)
                g = exp.(gm.gamma)
                
                th, v = newton_raphson_scheme(b, g, gm.bsh, gm.gsh, data.p[id.ns,i],
                    data.q[id.pq,i], data.v[id.pv,i], data.th[id.slack,i],
                    mat, id, Niter = Niter, const_jac = const_jac)
                p_est, q_est = v2s_map(b, g, gm.bsh, gm.gsh, v, th, mat, id)
                
                loss_vth += sum(abs, th[id.ns] - data.th[id.ns,i]) + sum(abs, v[id.pq] - data.v[id.pq,i])
                loss_pq += sum(abs, p_est[id.slack] - data.p[id.slack,i]) + sum(abs, q_est[id.pv] - data.q[id.pv,i])
            end
            dy = compare_params_2_admittance(gm.beta, gm.gamma, gm.bsh, gm.gsh,
                data.b, data.g, data.bsh, data.gsh, mat)
            println([e, loss_vth, loss_pq, dy])
        end
    end
    return nothing
end


function batch_pq_based_train!(
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
        grads = IdDict()
        grads[ps[2]] = zeros(size(ps[2]))
        grads[ps[4]] = zeros(size(ps[4]))
        grads[ps[1]] = zeros(size(ps[1]))
        grads[ps[3]] = zeros(size(ps[3]))
        gs = Zygote.Grads(grads, ps)
        Threads.@threads for i in 1:Nbatch
            gs .+= gradient(ps) do
                b = -exp.(gm.beta)
                g = exp.(gm.gamma)
                th, v = newton_raphson_scheme(b, g, gm.bsh, gm.gsh,
                    data.p[id.ns,i], data.q[id.pq,i], data.v[id.pv,i],
                    data.th[id.slack,i], mat, id, Niter = Niter,
                    const_jac = const_jac)
                    
                p_est, q_est = v2s_map(b, g, gm.bsh, gm.gsh, v, th, mat, id)
                
                return sum(abs, p_est[id.slack] - data.p[id.slack,i]) + sum(abs, q_est[id.pv] - data.q[id.pv,i])
                    + sum(abs, th[id.ns] - data.th[id.ns,i]) + sum(abs, v[id.pq] - data.v[id.pq,i])
            end
        end
        
        Flux.update!(opt, ps, gs)

        if(mod(e, Ninter) == 0)
            loss_vth = 0
            loss_pq = 0
            for i in 1:Nbatch
                b = -exp.(gm.beta)
                g = exp.(gm.gamma)
                
                th, v = newton_raphson_scheme(b, g, gm.bsh, gm.gsh, data.p[id.ns,i],
                    data.q[id.pq,i], data.v[id.pv,i], data.th[id.slack,i],
                    mat, id, Niter = Niter, const_jac = const_jac)
                p_est, q_est = v2s_map(b, g, gm.bsh, gm.gsh, v, th, mat, id)
                
                loss_vth += sum(abs, th[id.ns] - data.th[id.ns,i]) + sum(abs, v[id.pq] - data.v[id.pq,i])
                loss_pq += sum(abs, p_est[id.slack] - data.p[id.slack,i]) + sum(abs, q_est[id.pv] - data.q[id.pv,i])
            end
            dy = compare_params_2_admittance(gm.beta, gm.gamma, gm.bsh, gm.gsh,
                data.b, data.g, data.bsh, data.gsh, mat)
            println([e, loss_vth, loss_pq, dy])
        end
    end
    return nothing
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

    costh = cos.(dtheta) # dtheta or dtheta_ns ??????????????????????????????
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




