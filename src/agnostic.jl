function kalman_stat_estimation(
    data::SystemData,
    mat::Matrices,
    id::Indices,
    id_batch::Vector{Int64};
    Niter::Int64 = 10,
)
    # WARNING this is still in a "goofing around" stage
    B = mat.Bm * (-data.b .* mat.Bmt)
    dV = zeros(size(data.v,1))
    dth = zeros(size(data.v,1))
    obs = id.pv # here we assume that one observes just the generator buses
    nobs = setdiff(1:id.Nbus, obs)
    Nobs = length(obs)
    Nnobs = length(nobs)
    
    dV = data.v[:,1] .- 1.0
    dth = data.th[:,1] .- data.th[id.slack,1]
    
    dP = data.p[:,1]
    dQ = data.q[:,1] + data.bsh

    x = [dV[obs]; zeros(Nnobs); dth[obs]; zeros(Nnobs)]
    z = [dP[obs]; dQ[obs]; dV[obs]; dth[obs]];
    
    #v0 = zeros(id.Nbus) # TODO modify this
    #th0 = zeros(id.Nbus)
    
    Idobs = sparse(1:Nobs, 1:Nobs, ones(Nobs))
    HPth = - mat.Bm * (data.b .* mat.Bmt)
    #HQV = -mat.Bm * (data.b .* mat.Bmt) - sparse(1:id.Nbus, 1:id.Nbus, data.bsh)
    #HQV = (mat.Bout * (data.b .* mat.Bint) + mat.Bin * (data.b .* mat.Boutt)) -
    #    sparse(1:id.Nbus, 1:id.Nbus, mat.Bp * data.b + data.bsh)
    HQV = mat.Bm * (-data.b .* mat.Bmt) -
        2*sparse(1:id.Nbus, 1:id.Nbus, data.bsh)

    H = SparseMatrixCSC{Float64, Int64}([sparse([], [], [], Nobs, Nnobs+Nobs) HPth[obs,obs] HPth[obs,nobs];
            HQV[obs,obs] HQV[obs,nobs] sparse([], [], [], Nobs, Nnobs+Nobs);
            Idobs sparse([], [], [], Nobs, 2*Nnobs+Nobs);
            sparse([], [], [], Nobs, Nnobs+Nobs) Idobs sparse([], [], [], Nobs, Nnobs)]);
    
    alpha = 1
    P = sparse(1:2*id.Nbus, 1:2id.Nbus, [alpha*ones(Nobs); ones(Nnobs); alpha*ones(Nobs); ones(Nnobs)])
    R = sparse(1:4*Nobs, 1:4*Nobs, ones(4*Nobs))
    Id = sparse(1:2*id.Nbus, 1:2*id.Nbus, ones(2*id.Nbus))
    
    # cause left division seems not to be supported for sparse matrices
    H = Matrix{Float64}(H)
    Ht = Matrix{Float64}(sparse(H'))
    P = Matrix{Float64}(P)
    R = Matrix{Float64}(R)
    Id = Matrix{Float64}(Id)

    for k=1:Niter
        K = P * Ht / (H * P * Ht + R)
        x = x + K * (z - H * x)
        P = (Id - K * H) * P # assuming Q = 0
    end
    
    return 0
end
