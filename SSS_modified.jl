module SSS_modified

using DelimitedFiles

using DataStructures

using Random

using JuMP, Clp

using Plots

using StatsBase

using DPMMSubClusters

using Distances


dist_matrix = readdlm("diffchromall_CharCostFunction2.5.txt")

const MAX_NUM_CHILDREN = 10

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
	distance = dist_matrix[convert(Int64, x[1]), convert(Int64, y[1])]
	return distance
end

function distance_euclidean(x, y)
	return evaluate(Euclidean(), x, y)
end

saved_distances = zeros(0)

mutable struct Query
	focus :: Any
	radius :: Float64
end

mutable struct MultiFocalNode
	id 		:: Any
    foci        :: Vector{Any}
    children :: Vector{MultiFocalNode}
    radius :: Float64
    weights :: Vector{Float64}
end


function make_queries(query_dataset, query_radius)

	num_queries = size(query_dataset, 1)

	#query_foci = sample(query_dataset, num_queries, replace=false)

	test_query_foci_index = sample(1:num_queries, convert(Int64, floor(num_queries/2)), replace=false)

	query_radi = rand(0:query_radius, num_queries)

	#@assert issetequal(Set(query_foci), Set(query_dataset))

	training_query_dataset = Vector{Query}()
	testing_query_dataset = Vector{Query}()

	for i = 1:num_queries
		if i in test_query_foci_index
			push!(training_query_dataset, Query(query_dataset[i], query_radi[i]))
		else
			push!(testing_query_dataset, Query(query_dataset[i], query_radi[i]))
		end
	end

	

	return training_query_dataset, testing_query_dataset

end

function make_query_objects(query_dataset, query_radius)

	test_query_foci_index = sample(1:convert(Int64, floor(size(query_dataset,2)/2)), convert(Int64, floor(size(query_dataset,2)/2)), replace=false)

	test_query_foci = Vector{Any}()
	train_query_foci = Vector{Any}()

	for i=1:size(query_dataset,2)
		if i in test_query_foci_index
			push!(test_query_foci, query_dataset[:,i])
		else
			push!(train_query_foci, query_dataset[:,i])
		end
	end


	test_query_radi = rand(0:query_radius, size(test_query_foci, 1))

	test_queries = Vector{Query}()

	for i = 1:size(test_query_foci, 1)
		push!(test_queries, Query(test_query_foci[i], test_query_radi[i]))
	end

	train_query_radi = rand(0:query_radius, size(train_query_foci, 1))

	train_queries = Vector{Query}()

	for i = 1:size(train_query_foci, 1)
		push!(train_queries, Query(train_query_foci[i], train_query_radi[i]))
	end

	return test_queries, train_queries

end

function make_base_tree(dataset)
	multi_sss_children = Vector{MultiFocalNode}()
	for i = 1:size(dataset, 2)
		push!(multi_sss_children, MultiFocalNode(dataset[:,i],[dataset[:,i]], Vector{MultiFocalNode}(), 0, zeros(0)))
	end
	return MultiFocalNode(0,[0],multi_sss_children, 0, zeros(0))
end



function build_multi_ssstree(node)

	if size(node.children,1) < MAX_NUM_CHILDREN
		foci = Vector{Any}()
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

		foci = Vector{Any}()
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

		foci = Vector{Any}()
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

		foci = Vector{Any}()
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


function find_range(query, queue_normal, queue_optimized, result, search_distance, comparisons)
	w_tree = dequeue!(queue_optimized)
	nw_tree = dequeue!(queue_normal)

	point = query.focus
	range = query.radius
	query_dist = 0
	if nw_tree.foci[1] == 0.0
		for i = 1:size(nw_tree.children, 1)
			enqueue!(queue_normal, nw_tree.children[i])
			enqueue!(queue_optimized, w_tree.children[i])
		end
	else
		nw_weighted_distance = 0
		w_weighted_distance = 0
		temp_distances = zeros(0)
		if nw_tree.foci[1] == nw_tree.id 
			for i = 1:size(nw_tree.foci,1)
				distance = search_distance(nw_tree.foci[i], point)
				push!(temp_distances, distance)
				nw_weighted_distance += distance*nw_tree.weights[i]
				w_weighted_distance += distance*w_tree.weights[i]
				if nw_tree.id == nw_tree.foci[i]
					query_dist = distance
				end

			end
			global saved_distances = temp_distances
		else
			temp_distances = saved_distances
			for i = 1:size(nw_tree.foci,1)
				nw_weighted_distance += temp_distances[i]*nw_tree.weights[i]
				w_weighted_distance += temp_distances[i]*w_tree.weights[i]
				if nw_tree.id == nw_tree.foci[i]
					query_dist = temp_distances[i]
				end
			end
		end

		if nw_tree.foci[size(nw_tree.foci,1)] == nw_tree.id
			global saved_distances = zeros(0)
		end

		dist_to_query = query_dist

		if dist_to_query - range <= 0
			push!(result, nw_tree.id)
		end

		if nw_weighted_distance - nw_tree.radius > w_weighted_distance - w_tree.radius
			dist_to_point = nw_weighted_distance
			tree_radius = nw_tree.radius
		else	
			dist_to_point = w_weighted_distance
			tree_radius = w_tree.radius
		end


		if dist_to_point <= range + tree_radius + TOLERANCE
			for i = 1:size(nw_tree.children, 1)
				enqueue!(queue_normal, nw_tree.children[i])
				enqueue!(queue_optimized, w_tree.children[i])
			end
		end
	end
	if ! isempty(queue_normal)
		find_range(query, queue_normal, queue_optimized, result, search_distance, comparisons)
	else
		return result, comparisons()
	end
end


function find_range_linear_search(query, dataset, distance_function)
	linear_search_distance, liner_search_comparisons = counter(distance_function)

	result = Vector{Any}()
	for i=1:size(dataset, 2)
		if linear_search_distance(query.focus, dataset[:,i]) <= query.radius
			push!(result, dataset[:,i])
		end
	end
	return result, liner_search_comparisons
end

function search_tree(query, tree, dataset, distance_function)
	search_distance, search_comparisons = counter(distance_function)

	queue = Queue{MultiFocalNode}()

	enqueue!(queue, tree)

	result, comparisons = find_range(query, queue, Vector{Any}(), search_distance, search_comparisons)

	linear_search_result, c = find_range_linear_search(query, dataset, distance_function)

	@assert issetequal(Set(result), Set(linear_search_result))

	return result, comparisons

end

function search_tree(query, nw_tree, w_tree, dataset, distance_function)
	search_distance, search_comparisons = counter(distance_function)

	queue_normal = Queue{MultiFocalNode}()

	enqueue!(queue_normal, nw_tree)

	queue_optimized = Queue{MultiFocalNode}()

	enqueue!(queue_optimized, w_tree)

	result, comparisons = find_range(query, queue_normal, queue_optimized, Vector{Any}(), search_distance, search_comparisons)

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
	sum_comparisons_both = 0

	all_comparisons_weighted = Vector{Float64}()

	all_comparisons_non_weighted = Vector{Float64}()

	result_var_bedre = 0

	result_var_bedre_eller_lik = 0

	test_query_length = size(testing_queries, 1)

	for i = 1:test_query_length

		result_non_weighted, comparisons_non_weighted = search_tree(testing_queries[i], non_weighted_tree, dataset, distance_function)
		result_weighted, comparisons_weighted = search_tree(testing_queries[i], weighted_tree, dataset, distance_function)
		result_both, comparisons_both = search_tree(testing_queries[i], non_weighted_tree, weighted_tree, dataset, distance_function)

		sum_comparisons_weighted += comparisons_weighted
		sum_comparisons_non_weighted += comparisons_non_weighted
		sum_comparisons_both += comparisons_both


		push!(all_comparisons_non_weighted, comparisons_non_weighted)
		push!(all_comparisons_weighted, comparisons_weighted)

		if comparisons_non_weighted > comparisons_weighted
			result_var_bedre += 1
		end

		if comparisons_non_weighted >= comparisons_weighted
			result_var_bedre_eller_lik += 1
		end

		"""if size(result_weighted,1) > 0
			println("reuslt er mer enn null ")
			println(result_weighted)
			println(result_non_weighted)
		end"""

	end

	println(sum_comparisons_weighted/test_query_length)
	println(sum_comparisons_non_weighted/test_query_length)
	println(sum_comparisons_both/test_query_length)

	println(result_var_bedre/test_query_length)
	println(result_var_bedre_eller_lik/test_query_length)


	#savefig(plot(1:test_query_length, [all_comparisons_non_weighted, all_comparisons_weighted], title="Comparisons",label=["Non weighted" "Weighted"]), "plot.png")

	return sum_comparisons_weighted/test_query_length, sum_comparisons_non_weighted/test_query_length, sum_comparisons_both/test_query_length
end


function test_queries_dist_matrix(query_radius)

	global distance,comparisons = counter(distance_from_matrix)

	global distance_opt, comparisons_opt = counter(distance_from_matrix)

	dataset = transpose(1:floor(size(dist_matrix,1)*3/4))

	query_dataset = floor(size(dist_matrix,1)/2):size(dist_matrix,1)

	training_queries, testing_queries = make_queries(query_dataset, query_radius)

	#testing_queries = make_queries(query_dataset, 100, query_radius)

	#training_queries = make_queries(query_dataset, 100, query_radius)

	weighted_tree, non_weighted_tree = build_trees(dataset, training_queries)

	return test_queries(weighted_tree, non_weighted_tree, dataset, distance_from_matrix, testing_queries)
	
end

function test_queries_gaussian(tree_size, query_radius, num_training_queries, dimension)

	global distance,comparisons = counter(distance_euclidean)

	global distance_opt, comparisons_opt = counter(distance_euclidean)

	dataset, labeling, cluster_means = generate_gaussian_data(tree_size,dimension,4,100.0)

	query_dataset, labeling, cluster_means = generate_gaussian_data(num_training_queries,dimension,3,50.0)

	#testing_query_dataset, labeling, cluster_means = generate_gaussian_data(2000,dimension,20,100.0)

	testing_queries, training_queries = make_query_objects(query_dataset, query_radius)

	#training_queries = make_query_objects(training_query_dataset, query_radius)

	weighted_tree, non_weighted_tree = build_trees(dataset, training_queries)

	return test_queries(weighted_tree, non_weighted_tree, dataset, distance_euclidean, training_queries)
	
end

function plot_diffferent_data_sizes()
	w_results = Vector{Float64}()
	nw_results = Vector{Float64}()
	b_results = Vector{Float64}()

	io_w = open("plot_latex_datasize_w.txt", "w")
	io_nw = open("plot_latex_datasize_nw.txt", "w")



	#for i=[1000,5000,10000,15000,20000]
	for i=[1000,5000,7000,8000]
		w_result, nw_result, b_result = test_queries_gaussian(i, 25, 1000, 5)
		push!(w_results, w_result)
		push!(nw_results, nw_result)
		push!(b_results, b_result)
		#write(io_w, "($i,$w_result)")
		#write(io_nw, "($i,$nw_result)")
	end

	close(io_nw)
	close(io_w)

	println(w_results)
	println(nw_results)
	println(b_results)

	relation = b_results ./ nw_results

	savefig(plot([1000,5000,10000,20000], [relation], title="Comparisons",label=["Relation"]), "plot_datasize.png")


end



end
