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
