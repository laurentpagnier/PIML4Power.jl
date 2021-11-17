
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

function Newton_Raphson_scheme(
    b::Vector{Float64},
    g::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    p::Vector{Float64},
    q::Vector{Float64},
    vg::Vector{Float64},
    theta_slack::Float64,
    mat::Matrices,
    id::Indices;
    Niter = 3::Int64
)
    v = zeros(id.Nbus)
    th = zeros(id.Nbus)
    v = v + mat.pv2full * vg  + mat.pq2full * ones(length(id.pq)) 
    th = th + vec(mat.s2full * theta_slack)

    v_new = zeros(id.Nbus)
    th_new = zeros(id.Nbus)

    for i in 1:Niter
        dpq = dPQ(b, g, bsh, gsh, v, th, mat, id, p, q)
        jac = jacob(b, g, bsh, gsh, v, th, mat, id)
        x = jac \ dpq
        th_new = th - mat.ns2full * x[1:id.Nbus-1]
        v_new = v - mat.pq2full * x[id.Nbus:end]
        v = v_new
        th = th_new
    end
    
    return th, v
end


function loss(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    p::Vector{Float64},
    q::Vector{Float64},
    vg::Vector{Float64},
    thref::Vector{Float64},
    vref::Vector{Float64},
    pref::Vector{Float64},
    qref::Vector{Float64},
    theta_slack::Float64,
    mat::Matrices,
    id::Indices;
    Niter = 3::Int64
)
    b = -exp.(beta)
    g = exp.(gamma)
    th, v = Newton_Raphson_scheme(b, g, bsh, gsh, p, q, vg,
        theta_slack, mat, id, Niter = Niter)
    p_est, q_est = V2S(b, g, bsh, gsh, v, th, mat, id)

    return sum(abs.(th - thref)) + sum(abs.(v - vref)) +
        sum(abs.(p_est - pref)) + sum(abs.(q_est - qref)) 
end


function batch_training(
    beta::Vector{Float64},
    gamma::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    p::Matrix{Float64},
    q::Matrix{Float64},
    vg::Matrix{Float64},
    thref::Matrix{Float64},
    vref::Matrix{Float64},
    pref::Matrix{Float64},
    qref::Matrix{Float64},
    theta_slack::Vector{Float64},
    mat::Matrices,
    id::Indices;
    Niter = 3::Int64
)
    Nbatch = size(vg,2)
    grad = (zeros(length(beta)),zeros(length(beta)),zeros(length(gsh)),zeros(length(bsh)))
    for batch in 1:Nbatch
        g = gradient((beta, gamma, bsh, gsh) -> loss(beta, gamma, bsh, gsh, p[:,batch],
        q[:,batch], vg[:,batch], thref[:,batch], vref[:,batch], pref[:,batch], qref[:,batch], th_slack[batch], mat, id, Niter = Niter), beta, gamma, bsh, gsh)

        grad[1] .+= g[1] / Nbatch
        grad[2] .+= g[2] / Nbatch
        grad[3] .+= g[3] / Nbatch
        grad[4] .+= g[4] / Nbatch
    end
    return grad
end
