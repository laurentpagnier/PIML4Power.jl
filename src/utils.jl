export load_data, generate_full_line_list, generate_neighbour_list


function create_indices(slack::Int64, id_pv::Vector{Int64}, Nbus::Int64)
    ns = setdiff(collect(1:Nbus), slack) # indices of non-slack buses
    pq = setdiff(collect(1:Nbus), id_pv) # indices of PV buses (and slack)
    return Indices(slack, id_pv, pq, ns, Nbus)
end


function save_grid_params(filename::String)
    if isfile(filename)
        rm(filename)
    end
    h5write(filename, "/beta", beta)
    h5write(filename, "/gamma", gamma)
    h5write(filename, "/bsh", bsh)
    h5write(filename, "/gsh", gsh)
end


function load_data(filename::String)
    data = h5read(filename, "/")
    vref = data["V"]
    thref = data["theta"]
    pref = data["P"]
    qref = data["Q"]
    epsilon = Int64.(data["epsilon"])
    g_ref = vec(data["g"])
    obs = vec(Int64.(data["idgen"]))
    nobs = setdiff(1:maximum(epsilon), obs)
    b_ref = vec(data["b"])
    gsh_ref = vec(data["gsh"])
    bsh_ref = vec(data["bsh"])
    id_slack = Int64(data["id_slack"][1])
    id_pv = vec(Int64.(data["idgen"]))
    id = create_indices(id_slack, id_pv, size(vref,1))
    mat = create_incidence_matrices(epsilon, id)
    return vref, thref, pref, qref, epsilon, g_ref, gsh_ref, b_ref, bsh_ref, mat, id
end


function create_incidence_matrices(epsilon::Matrix{Int64}, id::Indices)
    Nline = size(epsilon,1)
    Nbus = maximum(epsilon)

    Bout = sparse(epsilon[:,1], 1:Nline, ones(Nline), Nbus, Nline)
    Bin = sparse(epsilon[:,2], 1:Nline, ones(Nline), Nbus, Nline)
    Bm = sparse([epsilon[:,1]; epsilon[:,2]], [1:Nline; 1:Nline],
        [-ones(Nline); ones(Nline)], Nbus, Nline)
    Bp = sparse([epsilon[:,1]; epsilon[:,2]], [1:Nline; 1:Nline],
        [ones(Nline); ones(Nline)], Nbus, Nline)

    temp = zeros(length(id.pq))
    for i in 1:length(id.pq)
    
        temp[i] = findall(id.ns .== id.pq[i])[1]
    end
    pq2ns = sparse(temp, 1:length(id.pq),
        ones(length(id.pq)), length(id.ns), length(id.pq))
    
    pv2full = sparse(id.pv, 1:length(id.pv), ones(length(id.pv)),
        id.Nbus, length(id.pv))
    s2full = sparse([id.slack], [1], [1.0], id.Nbus, 1)
    ns2full = sparse(id.ns, 1:length(id.ns), ones(length(id.ns)),
        id.Nbus, length(id.ns))
    pq2full = sparse(id.pq, 1:length(id.pq), ones(length(id.pq)),
        id.Nbus, length(id.pq))
    Ins = sparse(1:length(id.ns), 1:length(id.ns), ones(length(id.ns)))
    Ipq = sparse(1:length(id.pq), 1:length(id.pq), ones(length(id.pq)))
    
    Bp_ns = Bp[id.ns, :]
    Bp_pq = Bp[id.pq, :]
    Bm_ns = Bin[id.ns, :]
    Bin_ns = Bin[id.ns, :]
    Bin_pq = Bin[id.pq, :]
    Bout_ns = Bout[id.ns, :]
    Bout_pq = Bout[id.pq, :]

    Bmt = SparseMatrixCSC{Float64, Int64}(Bm')
    Bm_nst = SparseMatrixCSC{Float64, Int64}(Bm_ns')
    Bint = SparseMatrixCSC{Float64, Int64}(Bin')
    Bin_nst = SparseMatrixCSC{Float64, Int64}(Bin_ns')
    Bin_pqt = SparseMatrixCSC{Float64, Int64}(Bin_pq')
    Boutt = SparseMatrixCSC{Float64, Int64}(Bout')
    Bout_nst = SparseMatrixCSC{Float64, Int64}(Bout_ns')   
    Bout_pqt = SparseMatrixCSC{Float64, Int64}(Bout_pq')
    pq2nst = SparseMatrixCSC{Float64, Int64}(pq2ns')
    
    mat = Matrices(Bm, Bp, Bin, Bout, Bp_ns, Bp_pq, Bm_ns, Bin_ns,
        Bin_pq, Bout_ns, Bout_pq, Bmt, Bm_nst, Bint, Bin_nst, Bin_pqt,
        Boutt, Bout_nst, Bout_pqt, pq2ns, pq2nst, pv2full, s2full,
        ns2full, pq2full, Ins, Ipq)
    return mat
end


function generate_full_line_list(
    N::Int64
)
    id1 = Array{Int64}([])
    id2 = Array{Int64}([])
    for i = 1:N
        for j = i+1:N
            append!(id1, i)
            append!(id2, j)
        end
    end
    return [id1 id2]
end


function generate_neighbour_list(
    epsilon::Matrix{Int64},
    n::Int64,
)
    B = sparse([epsilon[:,1]; epsilon[:,2]], [1:size(epsilon,1); 1:size(epsilon,1)],
        [-ones(size(epsilon,1)); ones(size(epsilon,1))]);
    L0 = B * B'
    L = copy(L0)
    for i = 1:n-1
        L *= L0
    end
    
    id1 = Array{Int64}([])
    id2 = Array{Int64}([])
    for i = 1:length(L.colptr)-1
        println(i)
        ids = L.rowval[L.colptr[i]:L.colptr[i+1]-1]
        for j = 1:length(ids)
            if(ids[j] > i)
                append!(id1, i)
                append!(id2, ids[j])
            end
        end
    end
    return [id1 id2]
end
