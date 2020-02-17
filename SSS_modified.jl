module SSS_modified

using DelimitedFiles

using DataStructures

using Random

using JuMP, Clp

using Plots

using StatsBase


dist_matrix = readdlm("diffchromall_CharCostFunction2.5.txt")

const MAX_NUM_CHILDREN = 50

const QUERY_RADIUS = 20

const TOLERANCE = 0.00000001


function counter(f)
	count = 0

	function mapping(args...)
		count += 1
		f(args...)
	end
	extract() = count
	return mapping, extract
end

function distance_from_matrix(x, y)
	distance = dist_matrix[convert(Int64, x), convert(Int64, y)]
	return distance
end

saved_distances = zeros(0)

mutable struct Query
	focus :: Float64
	radius :: Float64
end

mutable struct MultiFocalNode
	id 		:: Int64
    foci        :: Vector{Float64}
    children :: Vector{MultiFocalNode}
    radius :: Float64
    weights :: Vector{Float64}
end

function make_queries(query_dataset, num_queries)

	query_foci = sample(query_dataset, num_queries, replace=false)

	query_radi = rand(0:QUERY_RADIUS, num_queries)

	queries = Vector{Query}()

	for i = 1:num_queries
		push!(queries, Query(query_foci[i], query_radi[i]))
	end

	return queries

end

function make_base_tree(dataset)
	multi_sss_children = Vector{MultiFocalNode}()
	for i = 1:size(dataset, 1)
		push!(multi_sss_children, MultiFocalNode(dataset[i],[dataset[i]], Vector{MultiFocalNode}(), 0, zeros(0)))
	end
	return MultiFocalNode(0,[0],multi_sss_children, 0, zeros(0))
end


function build_multi_ssstree(node)

	if size(node.children,1) < MAX_NUM_CHILDREN
		foci = zeros(0)
		for i = 1: size(node.children,1)
			push!(foci, node.children[i].foci[1])
		end

		for i = 1: size(node.children,1)
			weights = zeros(0)
			for j = 1:size(foci,1)
				if j == i
					push!(weights,1)
				else
					push!(weights,0)
				end
			end
			node.children[i].foci = foci
			node.children[i].weights = weights
		end
		radius = 0
		for i=1:size(node.children,1)
			dist = distance(node.id, node.children[i].id)
			if dist > radius
				radius = dist
			end
		end
		node.radius = radius
		node
	else

		list = node.children
		max = 0
		for x = 1:size(list,1) 
			for y = 1:size(list,1)
				dist =  distance(list[x].foci[1], list[y].foci[1])
				if dist>max
					max = dist
				end
			end
		end
		Ma = 0.4*max
		radius = 0
		if node.foci[1] == 0
			node.radius = 100000
		else
			node_index = 1
			for i=1:size(node.foci,1)
				if node.weights[i] == 1
					node_index = i
				end
			end
			for i = 1:size(list,1)
				dist = distance(node.foci[node_index], list[i].foci[1])
				if dist > radius
					radius = dist
				end
			end
			node.radius = radius
		end

		children_list = Vector{MultiFocalNode}()
		first_node = list[1]
		push!( children_list, first_node )

		add_new = 1


		for x = 2:size(list,1)
			nodex = list[x]
			add_new = 1
			pot_parent = MultiFocalNode(0,[0],zeros(0), 0, zeros(0))
			min_dist = 100000
			for y = 1:size(children_list,1)
				nodey = children_list[y]
				
				dist = distance(nodey.foci[1], nodex.foci[1])

				if dist < Ma
					if dist < min_dist
						min_dist = dist
						pot_parent = children_list[y]
					end
					add_new = 0
				end
			end

			if  add_new == 1
				new_node = nodex
				push!( children_list, new_node )
			else
				push!(pot_parent.children, nodex)
			end
			
		end

		foci = zeros(0)
		for i = 1: size(children_list,1)
			push!(foci, children_list[i].foci[1])
		end
		

		for i = 1: size(children_list,1)
			weights = zeros(0)
			for j = 1:size(foci,1)
				if j == i
					push!(weights,1)
				else
					push!(weights,0)
				end
			end
			children_list[i].foci = foci
			children_list[i].weights = weights
		end
		
	

		node.children = Vector{MultiFocalNode}()

		for i = 1:size(children_list,1)
			push!(node.children, build_multi_ssstree(children_list[i]))
		end

		node

	end
end


function coeff(PF, QF)

    mod = Model(with_optimizer(Clp.Optimizer, LogLevel = 0))
    PF = transpose(PF)
    n, m = size(PF)
    k = 1

    AvgQF = QF

    @variable(mod, u[1:k,1:m] ≥ 0)
    @variable(mod, v[1:k,1:m] ≥ 0)
    @variable(mod, r[1:k])

    # Maximize the sum of lower bounds. As the facets are independent, this
    # will optimize the components individually.

    @objective(mod, Max,
        sum(
            sum(AvgQF[i] * (u[f,i] - v[f,i]) for i = 1:m) - r[f]
        for f = 1:k)
    )

    for f = 1:k, i = 1:n
        @constraint(mod, sum(PF[i,j] * (u[f,j] - v[f,j]) for j = 1:m) ≤ r[f])
    end

    for f = 1:k
        @constraint(mod, sum(u[f,i] + v[f,i] for i = 1:m) == 1)
    end



    optimize!(mod)

    rad = [value(r[i]) for i=1:k]
    wts = [value(u[f,i]) - value(v[f,i]) for f=1:k, i=1:m]



    return wts, rad[1]

end


function build_multi_ssstree_optimized(node, z, training_queries)

	if size(node.children,1) < MAX_NUM_CHILDREN

		X = zeros(Float64, size(node.foci,1), size(node.children,1))

		for i=1:size(node.foci,1)
			for j=1:size(node.children, 1)
				X[i,j] = distance_opt(node.foci[i], node.children[j].id)	
			end
		end
		optimized_weights, r = coeff(X, z)
		node.weights = vec(optimized_weights)
		node.radius = r

		foci = zeros(0)
		for i = 1: size(node.children,1)
			push!(foci, node.children[i].foci[1])
		end

		for i = 1: size(node.children,1)
			weights = zeros(0)
			for j = 1:size(foci,1)
				if j == i
					push!(weights,1)
				else
					push!(weights,0)
				end
			end
			node.children[i].foci = foci
			node.children[i].weights = weights
		end
		node
	else

		list = node.children
		max = 0
		for x = 1:size(list,1) 
			for y = 1:size(list,1)
				dist =  distance_opt(list[x].foci[1], list[y].foci[1])
				if dist>max
					max = dist
				end
			end
		end
		Ma = 0.4*max
		radius = 0
		if node.foci[1] == 0
			node.radius = 100000
			node.weights = [1]
		end

		children_list = Vector{MultiFocalNode}()
		first_node = list[1]
		push!( children_list, first_node )

		add_new = 1

		for x = 2:size(list,1)
			nodex = list[x]
			add_new = 1
			pot_parent = MultiFocalNode(0,[0],zeros(0), 0, zeros(0))
			min_dist = 100000
			for y = 1:size(children_list,1)
				nodey = children_list[y]
				
				dist = distance(nodey.foci[1], nodex.foci[1])

				if dist < Ma
					if dist < min_dist
						min_dist = dist
						pot_parent = children_list[y]
					end
					add_new = 0
				end
			end

			if  add_new == 1
				new_node = nodex
				push!( children_list, new_node )
			else
				push!(pot_parent.children, nodex)
			end
			
		end

		foci = zeros(0)
		for i = 1: size(children_list,1)
			push!(foci, children_list[i].foci[1])
		end

		if node.id != 0
			X = zeros(Float64, size(node.foci,1), size(node.children,1))

			for i=1:size(node.foci,1)
				for j=1:size(node.children, 1)
					X[i,j] = distance_opt(node.foci[i], node.children[j].id)	
				end
			end
			optimized_weights, r = coeff(X, z)
			node.weights = vec(optimized_weights)
			node.radius = r
		end


		Z = zeros(0)

		for i=1:size(foci,1)
			sum_dist = 0
			for j=1:size(training_queries, 1)
				sum_dist += distance_opt(foci[i], training_queries[j].focus)
			end
			avg_dist = sum_dist/size(training_queries, 1)
			push!(Z, avg_dist)
		end

		

		for i = 1: size(children_list,1)
			children_list[i].foci = foci
		end
		
	

		node.children = Vector{MultiFocalNode}()


		for i = 1:size(children_list,1)
			push!(node.children, build_multi_ssstree_optimized(children_list[i], Z, training_queries))
		end

		node

	end
end


function find_range(query, queue, result, search_distance, comparisons)
	tree = dequeue!(queue)
	point = query.focus
	range = query.radius
	query_dist = 0
	if tree.foci[1] == 0.0
		for i = 1:size(tree.children, 1)
			enqueue!(queue, tree.children[i])
		end
	else
		weighted_distance = 0
		temp_distances = zeros(0)
		if tree.foci[1] == tree.id 
			for i = 1:size(tree.foci,1)
				distance = search_distance(tree.foci[i], point)
				push!(temp_distances, distance)
				weighted_distance += distance*tree.weights[i]
				if tree.id == tree.foci[i]
					query_dist = distance
				end

			end
			global saved_distances = temp_distances
		else
			temp_distances = saved_distances
			for i = 1:size(tree.foci,1)
				weighted_distance += temp_distances[i]*tree.weights[i]
				if tree.id == tree.foci[i]
					query_dist = temp_distances[i]
				end
			end
		end

		if tree.foci[size(tree.foci,1)] == tree.id
			global saved_distances = zeros(0)
		end

		dist_to_query = query_dist

		if dist_to_query - range <= 0
			push!(result, tree.id)
		end


		dist_to_point = weighted_distance

		if dist_to_point <= range + tree.radius + TOLERANCE
			for i = 1:size(tree.children, 1)
				enqueue!(queue, tree.children[i])
			end
		end
	end
	if ! isempty(queue)
		find_range(query, queue, result, search_distance, comparisons)
	else
		return result, comparisons()
	end
end


function find_range_linear_search(query, dataset, distance_function)
	linear_search_distance, liner_search_comparisons = counter(distance_function)

	result = zeros(0)
	for i=1:size(dataset, 1)
		if linear_search_distance(query.focus, i) <= query.radius
			push!(result, i)
		end
	end
	return result, liner_search_comparisons
end

function search_tree(query, tree, dataset, distance_function)
	search_distance, search_comparisons = counter(distance_function)

	queue = Queue{MultiFocalNode}()

	enqueue!(queue, tree)

	result, comparisons = find_range(query, queue, zeros(0), search_distance, search_comparisons)

	linear_search_result, c = find_range_linear_search(query, dataset, distance_function)

	@assert issetequal(Set(result), Set(linear_search_result))

	return result, comparisons

end

function build_trees(dataset, training_queries)
	
	weigthed_base_tree = make_base_tree(dataset)
	non_weigthed_base_tree = make_base_tree(dataset)

	weighted_tree = build_multi_ssstree_optimized(weigthed_base_tree, Vector{Float64}(), training_queries)
	non_weighted_tree = build_multi_ssstree(non_weigthed_base_tree)

	return weighted_tree, non_weighted_tree
end

function test_queries(weighted_tree, non_weighted_tree, dataset, distance_function, testing_queries)

	sum_comparisons_weighted = 0
	sum_comparisons_non_weighted = 0

	all_comparisons_weighted = Vector{Float64}()

	all_comparisons_non_weighted = Vector{Float64}()

	result_var_bedre = 0

	result_var_bedre_eller_lik = 0

	test_query_length = size(testing_queries, 1)

	for i = 1:test_query_length

		result_non_weighted, comparisons_non_weighted = search_tree(testing_queries[i], non_weighted_tree, dataset, distance_function)
		result_weighted, comparisons_weighted = search_tree(testing_queries[i], weighted_tree, dataset, distance_function)

		sum_comparisons_weighted += comparisons_weighted
		sum_comparisons_non_weighted += comparisons_non_weighted

		push!(all_comparisons_non_weighted, comparisons_non_weighted)
		push!(all_comparisons_weighted, comparisons_weighted)

		if comparisons_non_weighted > comparisons_weighted
			result_var_bedre += 1
		end

		if comparisons_non_weighted >= comparisons_weighted
			result_var_bedre_eller_lik += 1
		end

	end

	println(sum_comparisons_weighted/test_query_length)
	println(sum_comparisons_non_weighted/test_query_length)

	savefig(plot(1:test_query_length, [all_comparisons_non_weighted, all_comparisons_weighted], title="Comparisons",label=["Non weighted" "Weighted"]), "plot.png")

	return result_var_bedre_eller_lik/test_query_length
end


function test_queries_dist_matrix()

	global distance,comparisons = counter(distance_from_matrix)

	global distance_opt, comparisons_opt = counter(distance_from_matrix)

	dataset = 1:size(dist_matrix,1)/2

	query_dataset = 1:size(dist_matrix,1)

	testing_queries = make_queries(query_dataset, 100)

	training_queries = make_queries(query_dataset, 100)

	weighted_tree, non_weighted_tree = build_trees(dataset, training_queries)

	test_queries(weighted_tree, non_weighted_tree, dataset, distance_from_matrix, testing_queries)
	
end

test_queries_dist_matrix()

end
