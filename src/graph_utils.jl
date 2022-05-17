function get_line_graph(
    epsilon::Matrix{Int64} # list of the lines
    )
    # The line graph of an undirected graph G is another graph L(G) that
    # represents the adjacencies between edges of G.
    
    # lines are labelled from 1 to Nline
    line_epsilon = Int64.(zeros(0,2))
    for k=1:size(epsilon,1)
        id1 = epsilon[k,1]
        id2 = epsilon[k,2]
        # find which lines share the one bus with line k
        list = findall((epsilon[:,1] .== id1) .| (epsilon[:,2] .== id2))
        #list = setdiff(list, k) # remove line k from the list
        for m=1:length(list)
            if(k < list[m])
                line_epsilon = [line_epsilon; k list[m]]
            end
        end
    end
    return line_epsilon
end
