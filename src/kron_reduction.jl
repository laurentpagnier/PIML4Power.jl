export kron_reduction, build_admittance_matrix

function kron_reduction(
    Y::SparseMatrixCSC{ComplexF64, Int64},
    obs::Vector{Int64},
    nobs::Vector{Int64};
    alpha::Float64 = 1E-1,
)
    # stardard way to get a reduced equivalent model, see
    # J. Grainger & W. Stevenson, "Power System Analysis", 1994
    # or google "Kron reduction" 
    Nobs = length(obs)
    Nnobs = length(nobs)
    Y11 = Y[obs, obs]
    Y12 = Y[obs, nobs]
    Y21 = Y[nobs, obs]
    Y22 = Y[nobs, nobs]
    Yr = Y11 - Y12 * (Y22 \ Matrix{ComplexF64}(Y21)) 
    thres = alpha * minimum(abs.(Y.nzval))
    epsilon_r = zeros(0,2)
    y_r = Vector{ComplexF64}([])
    for i in 1:Nobs
        for j in i+1:Nobs
            if(abs(Yr[i,j]) > thres)
                epsilon_r = vcat(epsilon_r, [i j])
                append!(y_r, -Yr[i, j])
            end
        end
    end
    epsilon_r = Int64.(epsilon_r)
    Nline = size(epsilon_r, 1)
    id2 = [epsilon_r[:,1]; epsilon_r[:,2]]
    id1 = [1:Nline; 1:Nline]
    values = [ones(Nline); -ones(Nline)]
    Bm = sparse(id1, id2, values, size(epsilon_r,1), Nobs, Nline)
    ysh_r = diag(Yr) - Vector{ComplexF64}(diag(Bm' * sparse(1:Nline, 1:Nline, y_r) * Bm))
    return  y_r, ysh_r, epsilon_r
end


function contribution_of_nobs_buses(
    Y::SparseMatrixCSC{ComplexF64, Int64},
    obs::Vector{Int64},
    nobs::Vector{Int64},
    th::Matrix{Float64},
    v::Matrix{Float64},
    p::Matrix{Float64},
    q::Matrix{Float64},
)
    # this function gives the contribution of non-observersed (i.e. reduced)
    # buses to the effective power injections in the Kron-reduced equivalent
    # network. It is what the NN should learn in our hybrid scheme.
    Nobs = length(obs)
    Nnobs = length(nobs)
    Y12 = Y[obs, nobs]
    Y22 = Y[nobs, nobs]
    
    V = v .* exp.(im .* th)
    Iu = conj((p[nobs,:] + im * q[nobs,:]) ./ V[nobs,:])
    S = V[obs,:] .* conj(Y12 * (Y22 \ Iu))
    return  real(S), imag(S)
end


function build_admittance_matrix(
    b::Vector{Float64},
    g::Vector{Float64},
    bsh::Vector{Float64},
    gsh::Vector{Float64},
    epsilon::Matrix{Int64}
)
    Nline = size(epsilon,1)
    Nbus = maximum(epsilon)
    Bm = sparse([epsilon[:,1]; epsilon[:,2]], [1:Nline; 1:Nline],
        [-ones(Nline); ones(Nline)], Nbus, Nline)
    return Bm * sparse(1:Nline, 1:Nline, g+im*b) * Bm' + 
        sparse(1:Nbus, 1:Nbus, gsh+im*bsh)
end


function compute_power_flows(
    b::Vector{Float64},
    g::Vector{Float64},
    th::Matrix{Float64},
    v::Matrix{Float64},
    mat::Matrices,
)
    Nbatch = size(th,2)
    Nline = size(mat.Bin,2)
    pij = zeros(Nline,Nbatch)
    qij = zeros(Nline,Nbatch)
    pji = zeros(Nline,Nbatch)
    qji = zeros(Nline,Nbatch)
    
    Threads.@threads for i=1:Nbatch
        a, b, c, d = compute_power_flows(b, g, th[:,i], v[:,i], mat)
        pij[:,i] = a
        qij[:,i] = b
        pji[:,i] = c
        qji[:,i] = d
    end
        
    return pij, qij, pji, qji
end


function compute_power_flows(
    b::Vector{Float64},
    g::Vector{Float64},
    th::Vector{Float64},
    v::Vector{Float64},
    mat::Matrices,
)
    dth = mat.Bmt * th;
    costh = cos.(dth)
    sinth = sin.(dth)

    pij = g .* ((mat.Bint * v).^2 - (mat.Bint * v) .* (mat.Boutt * v) .* costh) -
        b .* (mat.Bint * v) .* (mat.Boutt * v) .* sinth
    qij = b .* (-(mat.Bint * v).^2 + (mat.Bint * v) .* (mat.Boutt * v) .* costh) -
        g .* (mat.Bint * v) .* (mat.Boutt * v) .* sinth
    pji = g .* ((mat.Boutt * v).^2 - (mat.Bint * v) .* (mat.Boutt * v) .* costh) +
        b .* (mat.Bint * v) .* (mat.Boutt * v) .* sinth
    qji = b .* (-(mat.Boutt * v).^2 + (mat.Bint * v) .* (mat.Boutt * v) .* costh) +
        g .* (mat.Bint * v) .* (mat.Boutt * v) .* sinth
        
    return pij, qij, pji, qji
end


function build_PTDF_matrix(b::Vector{Float64}, mat::Matrices)
    # shunt suscpetances are neglected
    # define the susceptance matrix
    B = mat.Bm * (b .* mat.Bmt) # should be -b 
    # compute its pseudo-inverse
    eigs = eigen(Matrix{Float64}(L))
    v = eigs.values
    v[1:end-1] .= 1 ./ v[1:end-1]
    Bt = eigs.vectors * (eigs.values .* eigs.vectors')
    # return the pdtf mat
    return b .* mat.Bmt * Bt * v # should be -b be cancel out with the previous one
end
