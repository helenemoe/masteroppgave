module multi_sss_tree

export main, main2

using DelimitedFiles

using DataStructures

using Random

using JuMP, Clp

dist_matrix = readdlm("diffchromall_CharCostFunction2.5.txt")

const N = 4200

const MAX_NUM_CHILDREN = 10

const TEST_DATASET_SIZE = N/2

const TRAINING_QUERY_LENGTH = 100

const TEST_QUERY_LENGTH = 1000

const TEST_QUERY_RADIUS = 20

const TOLERANCE = 0.00000001

#dist_matrix = [[0 1 2 3 4 5 6]; [1 0 1 2 3 4 5]; [2 1 0 1 2 3 4]; [3 2 1 0 1 2 3]; [4 3 2 1 0 1 2]; [5 4 3 2 1 0 1]; [6 5 4 3 2 1 0]]

#dist_matrix_test = dist_matrix[1:3,1:3]

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

distance,comparisons = counter(distance_from_matrix)

distance_opt, comparisons_opt = counter(distance_from_matrix)

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


function make_training_queries()

	training_query_foci = rand(1:N, TRAINING_QUERY_LENGTH)

	training_queries = Vector{Query}()

	for i = 1:TRAINING_QUERY_LENGTH
		push!(training_queries, Query(training_query_foci[i], 0))
	end

	return training_queries

end

training_queries = make_training_queries()

function make_base_tree()
	multi_sss_children = Vector{MultiFocalNode}()
	for i = 1:TEST_DATASET_SIZE
		push!(multi_sss_children, MultiFocalNode(i,[i], Vector{MultiFocalNode}(), 0, zeros(0)))
	end
	return MultiFocalNode(0,[0],multi_sss_children, 0, zeros(0))
end

function make_test_queries()
	test_query_foci = rand(1:N, TEST_QUERY_LENGTH)

	test_query_radi = rand(0:TEST_QUERY_RADIUS, TEST_QUERY_LENGTH)

	test_queries = Vector{Query}()

	for i = 1:TEST_QUERY_LENGTH
		push!(test_queries, Query(test_query_foci[i], test_query_radi[i]))
	end

	return test_queries
end



function build_multi_ssstree(node)

	if size(node.children,1) < MAX_NUM_CHILDREN
	#if size(node.children,1) < 1

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
		#Ma = max
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


function build_multi_ssstree_optimized(node, z)

	if size(node.children,1) < MAX_NUM_CHILDREN
	#if size(node.children,1) < 1

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
		#Ma = max
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
			for j=1:TRAINING_QUERY_LENGTH
				sum_dist += distance_opt(foci[i], training_queries[j].focus) #- training_queries[j].radius
			end
			avg_dist = sum_dist/TRAINING_QUERY_LENGTH
			push!(Z, avg_dist)
		end

		

		for i = 1: size(children_list,1)
			children_list[i].foci = foci
			#children_list[i].weights = weights
		end
		
	

		node.children = Vector{MultiFocalNode}()


		for i = 1:size(children_list,1)
			push!(node.children, build_multi_ssstree_optimized(children_list[i], Z))
		end

		node

	end
end


function print_tree(tree)
	for i=1:size(tree.children,1)
		if tree.children[i].radius != 0
			print(tree.children[i].foci)
			print(tree.children[i].weights)
			print(" radius: ")
			print(tree.children[i].radius)
			println("Children")
			print_tree(tree.children[i])
		end

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


function find_range_linear_search(query)
	linear_search_distance, liner_search_comparisons = counter(distance_from_matrix)

	result = zeros(0)
	for i=1:TEST_DATASET_SIZE
		if linear_search_distance(query.focus, i) <= query.radius
			push!(result, i)
		end
	end
	return result, liner_search_comparisons
end

function search_tree(query, tree)
	search_distance, search_comparisons = counter(distance_from_matrix)

	queue = Queue{MultiFocalNode}()

	enqueue!(queue, tree)

	result, comparisons = find_range(query, queue, zeros(0), search_distance, search_comparisons)

	linear_search_result, c = find_range_linear_search(query)

	@assert issetequal(Set(result), Set(linear_search_result))

	return result, comparisons

end

function build_trees_and_test_queries()

	test_queries = make_test_queries()
	
	weigthed_base_tree = make_base_tree()
	non_weigthed_base_tree = make_base_tree()

	weighted_tree = build_multi_ssstree_optimized(weigthed_base_tree, Vector{Float64}())
	non_weighted_tree = build_multi_ssstree(non_weigthed_base_tree)

	sum_comparisons_weighted = 0
	sum_comparisons_non_weighted = 0

	result_var_bedre = 0

	result_var_bedre_eller_lik = 0

	for i = 1:TEST_QUERY_LENGTH

		result_non_weighted, comparisons_non_weighted = search_tree(test_queries[i], non_weighted_tree)
		result_weighted, comparisons_weighted = search_tree(test_queries[i], weighted_tree)

		sum_comparisons_weighted += comparisons_weighted
		sum_comparisons_non_weighted += comparisons_non_weighted

		if comparisons_non_weighted > comparisons_weighted
			result_var_bedre += 1
		end

		if comparisons_non_weighted >= comparisons_weighted
			result_var_bedre_eller_lik += 1
		end

	end

	println(sum_comparisons_weighted/TEST_QUERY_LENGTH)
	println(sum_comparisons_non_weighted/TEST_QUERY_LENGTH)

	return result_var_bedre_eller_lik/TEST_QUERY_LENGTH

end

function main()
	avg = 0
	for i=1:10
		avg += build_trees_and_test_queries()
		println("hei")
	end
	println(avg/10)


end

function main2(point, radius)

	test_query = Query(point, radius)
	
	weigthed_base_tree = make_base_tree()
	non_weigthed_base_tree = make_base_tree()

	weighted_tree = build_multi_ssstree_optimized(weigthed_base_tree, Vector{Float64}())
	non_weighted_tree = build_multi_ssstree(non_weigthed_base_tree)

	sum_comparisons_weighted = 0
	sum_comparisons_non_weighted = 0

	println("forste ")

	result_non_weighted, comparisons_non_weighted = search_tree(test_query, non_weighted_tree)

	println("haloo")

	result_weighted, comparisons_weighted = search_tree(test_query, weighted_tree)

	println("hei")

	

	sum_comparisons_weighted += comparisons_weighted
	sum_comparisons_non_weighted += comparisons_non_weighted

	println(sum_comparisons_weighted/TEST_QUERY_LENGTH)
	println(sum_comparisons_non_weighted/TEST_QUERY_LENGTH)

end


end
