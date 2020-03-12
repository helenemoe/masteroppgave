module SSS_modified

using DelimitedFiles

using DataStructures

using Random

using JuMP, Clp

using Plots

using StatsBase

using DPMMSubClusters

using Distances

using Distributions

using GaussianMixtures

using BSON


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

	query_radi = query_radius

	#@assert issetequal(Set(query_foci), Set(query_dataset))

	training_query_dataset = Vector{Query}()
	testing_query_dataset = Vector{Query}()

	for i = 1:num_queries
		if i in test_query_foci_index
			push!(training_query_dataset, Query(query_dataset[i], query_radi))
		else
			push!(testing_query_dataset, Query(query_dataset[i], query_radi))
		end
	end

	

	return training_query_dataset, testing_query_dataset

end

function make_queries(query_dataset, query_radius, num_queries)

	len_data = size(query_dataset, 1)
	#query_foci = sample(query_dataset, num_queries, replace=false)

	query_foci_index = sample(1:len_data, num_queries, replace=false)

	query_radi = query_radius

	#@assert issetequal(Set(query_foci), Set(query_dataset))

	query_dataset = Vector{Query}()

	for i=1:size(query_foci_index,1)
		push!(query_dataset, Query(query_foci_index[i], query_radi))
	end


	return query_dataset[1:convert(Int64,floor(num_queries/2))], query_dataset[convert(Int64,floor(num_queries/2)):num_queries]

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


	test_query_radi = query_radius

	test_query_radi = test_query_radi

	test_queries = Vector{Query}()

	for i = 1:size(test_query_foci, 1)
		push!(test_queries, Query(test_query_foci[i], test_query_radi))
	end

	train_query_radi = query_radius

	train_queries = Vector{Query}()

	for i = 1:size(train_query_foci, 1)
		push!(train_queries, Query(train_query_foci[i], train_query_radi))
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




function find_range_inner(query, queue, result, search_distance, comparisons)
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
		#println("tree.foci[1] $(tree.foci[1])")
		#println("tree.id $(tree.id)")
		if tree.foci[1] == tree.id
			#println("gikk inn")
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
				#println(temp_distances)
				#println(tree.weights)
				#println(i)
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

		if dist_to_query - range <= 0.0 + TOLERANCE
			push!(result, tree.id)
		end

		dist_to_point = weighted_distance

		#if true
		if dist_to_point <= range + tree.radius + TOLERANCE
			for i = 1:size(tree.children, 1)
				enqueue!(queue, tree.children[i])
			end
		end
	end
	"""
	if ! isempty(queue)
		find_range(query, queue, result, search_distance, comparisons)
	else
		return result, comparisons()
	end
	"""
	return queue, result, comparisons()
end


function find_range_inner(query, queue_normal, queue_optimized, result, search_distance, comparisons)
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

		#if true
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

		#if true
		if dist_to_point <= range + tree_radius + TOLERANCE
			for i = 1:size(nw_tree.children, 1)
				enqueue!(queue_normal, nw_tree.children[i])
				enqueue!(queue_optimized, w_tree.children[i])
			end
		end
	end
	"""
	if ! isempty(queue_normal)
		find_range(query, queue_normal, queue_optimized, result, search_distance, comparisons)
	else
		return result, comparisons()
	end
	"""
	return queue_normal, queue_optimized, result, comparisons()
end

function find_range(query, queue, result, search_distance, comparisons)
	comp = 0
	while ! isempty(queue)
		queue, result, comp = find_range_inner(query, queue, result, search_distance, comparisons)
	end
	return result, comp
end

function find_range(query, queue_normal, queue_optimized, result, search_distance, comparisons)
	comp = 0
	while ! isempty(queue_normal)
		queue_normal, queue_optimized, result, comp = find_range_inner(query, queue_normal, queue_optimized, result, search_distance, comparisons)
	end
	return result, comp
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

	@assert Set(result) ==  Set(linear_search_result)

	#@assert issetequal(Set(result), Set(linear_search_result))

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

	avg_restult = 0

	test_query_length = size(testing_queries, 1)

	for i = 1:test_query_length

		result_non_weighted, comparisons_non_weighted = search_tree(testing_queries[i], non_weighted_tree, dataset, distance_function)
		result_weighted, comparisons_weighted = search_tree(testing_queries[i], weighted_tree, dataset, distance_function)
		result_both, comparisons_both = search_tree(testing_queries[i], non_weighted_tree, weighted_tree, dataset, distance_function)

		sum_comparisons_weighted += comparisons_weighted
		sum_comparisons_non_weighted += comparisons_non_weighted
		sum_comparisons_both += comparisons_both
		avg_restult  += size(result_weighted,1)


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

	println("result:")
	println(avg_restult/test_query_length)

	println(result_var_bedre/test_query_length)
	println(result_var_bedre_eller_lik/test_query_length)


	#savefig(plot(1:test_query_length, [all_comparisons_non_weighted, all_comparisons_weighted], title="Comparisons",label=["Non weighted" "Weighted"]), "plot.png")

	return sum_comparisons_weighted/test_query_length, sum_comparisons_non_weighted/test_query_length, sum_comparisons_both/test_query_length, avg_restult/test_query_length
end

function find_points(number, amount)
	list = zeros(0)
	point = number
	hh = dist_matrix[point,:]
	for i=1:size(dist_matrix,1)
		if size(list,1)<amount
			push!(list, i)
		else
			po = hh[i]
			lol = false
			ll = 0
			lll = 0
			for j=1:size(list,1)
				if po<hh[convert(Int64,list[j])]
					lol = true
					if list[j]>ll
						ll=j
						lll = hh[convert(Int64,list[j])]
					end
				end
			end
			if lol == true
				deleteat!(list, ll)
				push!(list, i)
			end
		end
	end
	return list
end

function find_points(number, amount, distance_function, data)
	list = Array{Float64}(undef, size(data, 1),amount)
	point = data[:,number]

	distance,comparisons = counter(distance_function)

	for i=1:size(data,2)
		if i<=amount
			list[:,i] = data[:,i]
		else
			add_to_list = false
			change_distance = 0
			change_point = 0
			dist_i = distance(point, data[:,i])
			for j=1:size(list,2)
				dist_j = distance(point, list[:,j])
				if dist_i<dist_j
					add_to_list = true
					if dist_j>change_distance
						change_point=j
						change_distance = dist_j
					end
				end
			end
			if add_to_list == true
				list[:,change_point] = data[:,i]
			end
		end
	end
	return list
end

function find_highest()
	max_sum = 0
	max_i = 0
	for i = 1:size(dist_matrix,1)
		if sum(dist_matrix[:,i]) > max_sum
			max_sum = sum(dist_matrix[:,i])
			max_i = i
		end
	end 

	@assert max_i > 0

	return max_i
end

function test_queries_dist_matrix(query_radius)

	global distance,comparisons = counter(distance_from_matrix)

	global distance_opt, comparisons_opt = counter(distance_from_matrix)

	dataset = transpose(1:floor(size(dist_matrix,1)*3/4))

	#query_dataset = vcat(find_points(10, 100), find_points(3000, 50))
	point = rand(1:size(dist_matrix,1), 1)

	query_dataset = find_points(find_highest(), 1000) 

	#training_queries, testing_queries = make_queries(query_dataset, query_radius)


	#testing_queries = make_queries(dataset, 100, query_radius)

	#training_queries = make_queries(query_dataset, 100, query_radius)

	training_queries, testing_queries = make_queries(1:size(dist_matrix,1), query_radius, 100)

	weighted_tree, non_weighted_tree = build_trees(dataset, training_queries)

	bson("test.bson", Dict(:a => weighted_tree, :b => non_weighted_tree))

	return test_queries(weighted_tree, non_weighted_tree, dataset, distance_from_matrix, testing_queries)
	
end

function test_queries_dist_matrix(query_radius, num_times)

	global distance,comparisons = counter(distance_from_matrix)

	global distance_opt, comparisons_opt = counter(distance_from_matrix)

	dataset = transpose(1:floor(size(dist_matrix,1)*3/4))

	#query_dataset = vcat(find_points(10, 100), find_points(3000, 50))
	point = rand(1:size(dist_matrix,1), num_times)
	point2 = rand(1:size(dist_matrix,1), num_times)

	avg_w_result = 0
	avg_nw_result = 0
	avg_b_result = 0
	avg_num_result = 0

	
	for i=1:size(point,1)

		query_dataset = vcat(find_points(point[i], 50), find_points(point2[i], 50))

		training_queries, testing_queries = make_queries(query_dataset, query_radius)

		weighted_tree, non_weighted_tree = build_trees(dataset, training_queries)

		w_result, nw_result, b_result, num_result = test_queries(weighted_tree, non_weighted_tree, dataset, distance_from_matrix, testing_queries)

		avg_w_result += w_result
		avg_nw_result += nw_result
		avg_b_result += b_result
		avg_num_result += num_result

	end

	#training_queries, testing_queries = make_queries(query_dataset, query_radius)


	#testing_queries = make_queries(dataset, 100, query_radius)

	#training_queries = make_queries(query_dataset, 100, query_radius)

	training_queries, testing_queries = make_queries(1:size(dist_matrix,1), query_radius, 100)

	weighted_tree, non_weighted_tree = build_trees(dataset, training_queries)

	return avg_w_result/num_times, avg_nw_result/num_times, avg_b_result/num_times, avg_num_result/num_times
	
end


function test_queries_gaussian(tree_size, query_radius, num_training_queries, dimension)

	global distance,comparisons = counter(distance_euclidean)

	global distance_opt, comparisons_opt = counter(distance_euclidean)

	dataset, labeling, cluster_means = generate_gaussian_data(tree_size,dimension,4,100.0)

	dataset = convert(Array{Float64}, dataset)


	query_dataset, labeling, cluster_means = generate_gaussian_data(num_training_queries,dimension,1,5.0)

	query_dataset = convert(Array{Float64}, query_dataset)

	near_points = find_points(3, 100, distance_euclidean, dataset)


	#testing_query_dataset, labeling, cluster_means = generate_gaussian_data(2000,dimension,20,100.0)

	testing_queries, training_queries = make_query_objects(query_dataset, query_radius)

	#training_queries = make_query_objects(training_query_dataset, query_radius)

	weighted_tree, non_weighted_tree = build_trees(dataset, training_queries)

	return test_queries(weighted_tree, non_weighted_tree, dataset, distance_euclidean, testing_queries)
	
end

function test_queries_corel(query_radius, num_training_queries)

	global distance,comparisons = counter(distance_euclidean)

	global distance_opt, comparisons_opt = counter(distance_euclidean)

	dataset = read_corel_images()

	tree_dataset = dataset[:, 1:convert(Int64, floor(size(dataset,2)*0.5))]

	query_dataset = find_points(3, num_training_queries, distance_euclidean, dataset)

	#testing_query_dataset, labeling, cluster_means = generate_gaussian_data(2000,dimension,20,100.0)

	testing_queries, training_queries = make_query_objects(query_dataset, query_radius)

	#training_queries = make_query_objects(training_query_dataset, query_radius)

	weighted_tree, non_weighted_tree = build_trees(tree_dataset, training_queries)

	return test_queries(weighted_tree, non_weighted_tree, tree_dataset, distance_euclidean, testing_queries)
	
end

function test_queries_random_vectors(tree_size, query_radius, num_training_queries, dimension)

	global distance,comparisons = counter(distance_euclidean)

	global distance_opt, comparisons_opt = counter(distance_euclidean)

	dataset = rand(0:100, dimension)

	for i=1:tree_size-1
		dataset = hcat(dataset, rand(0:100, dimension))
	end

	query_dataset = rand(20:30, dimension)

	for i=1:num_training_queries-1
		query_dataset = hcat(query_dataset, rand(20:30, dimension))
	end

	tree_dataset = dataset

	#tree_dataset = dataset[:,1:convert(Int64, floor(size(dataset,2)/2))]

	#query_dataset = find_points(500, 100, distance_euclidean, dataset)


	testing_queries, training_queries = make_query_objects(query_dataset, query_radius)

	weighted_tree, non_weighted_tree = build_trees(tree_dataset, training_queries)


	return test_queries(weighted_tree, non_weighted_tree, tree_dataset, distance_euclidean, testing_queries)
	

end

function test_queries_gaussian_mixture(tree_size, query_radius, num_training_queries, dimension)

	global distance,comparisons = counter(distance_euclidean)

	global distance_opt, comparisons_opt = counter(distance_euclidean)

	dataset, labeling, cluster_means = generate_gaussian_data(tree_size,dimension,4,100.0)

	gm1 = rand(GMM, 4, 2, kind=:full, sep=13.0)
	data1 = rand(gm1, 4000)
	data1 = transpose(data1)

	gm2 = rand(GMM, 3, 2, kind=:full, sep=0.0)
	data2 = rand(gm2, 1000)
	data2 = transpose(data2)

	gm3 = rand(GMM, 5, 2, kind=:full, sep=25.0)
	data3 = rand(gm3, 3000)
	data3 = transpose(data3)

	data = hcat(data1,data2,data3)

	gmm = rand(GMM, 2, 2, kind=:full, sep=0.0)
	q_data = rand(gmm, 100)
	q_data = transpose(q_data)

	near_points = find_points(3, 100, distance_euclidean, data)

	near_points2 = find_points(6007, 100, distance_euclidean, data)

	near_points3 = hcat(near_points, near_points2)
	#testing_query_dataset, labeling, cluster_means = generate_gaussian_data(200,2,20,100.0)

	testing_queries, training_queries = make_query_objects(near_points, query_radius)

	#training_queries = make_query_objects(training_query_dataset, query_radius)

	weighted_tree, non_weighted_tree = build_trees(data, training_queries)

	return test_queries(weighted_tree, non_weighted_tree, data, distance_euclidean, testing_queries)
	
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


function plot_dist_matrix()
	w_results = Vector{Float64}()
	nw_results = Vector{Float64}()
	b_results = Vector{Float64}()

	percent_results = Vector{Float64}()

	io_w = open("plot_latex_chome_w.txt", "w")
	io_nw = open("plot_latex_chome_nw.txt", "w")
	io_b = open("plot_latex_chome_b.txt", "w")




	#for i=[1000,5000,10000,15000,20000]
	for i=[5,10, 15, 20, 25, 30]

		w_result, nw_result, b_result, num_result = test_queries_dist_matrix(i,10)
		println("num result: $num_result")

		w_result_f = 100*w_result/3150
		nw_result_f = 100*nw_result/3150
		b_result_f = 100*b_result/3150

		push!(w_results, w_result)
		push!(nw_results, nw_result)
		push!(b_results, b_result)

		push!(percent_results, num_result/3150)
		y_axis_f = 100*num_result/3150
		write(io_w, "($y_axis_f,$w_result_f)")
		write(io_nw, "($y_axis_f,$nw_result_f)")
		write(io_b, "($y_axis_f,$b_result_f)")
	end

	close(io_nw)
	close(io_w)
	close(io_b)

	println(w_results)
	println(nw_results)
	println(b_results)

	percent_w_results = w_results ./ 3150
	percent_nw_results = nw_results ./ 3150
	percent_b_results = b_results ./ 3150

	savefig(plot(percent_results, [percent_w_results, percent_nw_results, percent_b_results], title="Comparisons",label=["W" "NW" "B"]), "plot_chromecast.png")


end

function plot_random_vector(size_dataset)

	w_results = Vector{Float64}()
	nw_results = Vector{Float64}()
	b_results = Vector{Float64}()

	percent_results = Vector{Float64}()

	io_w = open("plot_latex_rand_w.txt", "w")
	io_nw = open("plot_latex_rand_nw.txt", "w")
	io_b = open("plot_latex_rand_b.txt", "w")




	#for i=[1000,5000,10000,15000,20000]
	for i=[110,140,200,300]

		w_result, nw_result, b_result, num_result = test_queries_random_vectors(size_dataset, i, 1000, 20)
		println("num result: $num_result")

		w_result_f = 100*w_result/size_dataset
		nw_result_f = 100*nw_result/size_dataset
		b_result_f = 100*b_result/size_dataset

		push!(w_results, w_result)
		push!(nw_results, nw_result)
		push!(b_results, b_result)

		push!(percent_results, num_result/size_dataset)
		y_axis_f = 100*num_result/size_dataset
		write(io_w, "($y_axis_f,$w_result_f)")
		write(io_nw, "($y_axis_f,$nw_result_f)")
		write(io_b, "($y_axis_f,$b_result_f)")
	end

	close(io_nw)
	close(io_w)
	close(io_b)

	println(w_results)
	println(nw_results)
	println(b_results)

	percent_w_results = w_results ./ size_dataset
	percent_nw_results = nw_results ./ size_dataset
	percent_b_results = b_results ./ size_dataset

	savefig(plot(percent_results, [percent_w_results, percent_nw_results, percent_b_results], title="Comparisons",label=["W" "NW" "B"]), "plot_random.png")


end

function read_corel_images()
	"""open("ColorMoments.asc") do color
		for line in eachline(color)
			println(line)
		end
	end"""

	color_moments = readdlm("ColorMoments.asc")

	color_moments = color_moments[:,2:size(color_moments,2)]

	color_moments = transpose(color_moments)

	return color_moments


end

function build_bson_corel()

	global distance,comparisons = counter(distance_euclidean)

	global distance_opt, comparisons_opt = counter(distance_euclidean)

	dataset = read_corel_images()

	tree_dataset = dataset[:, 1:convert(Int64, floor(size(dataset,2)*0.5))]

	query_dataset = find_points(20000, 100, distance_euclidean, dataset)

	testing_queries, training_queries = make_query_objects(query_dataset, 0)

	weighted_tree, non_weighted_tree = build_trees(tree_dataset, training_queries)

	bson("corel.bson", Dict(:w => weighted_tree, :nw => non_weighted_tree, :tq => testing_queries))

end

function build_bson_corel()

	global distance,comparisons = counter(distance_euclidean)

	global distance_opt, comparisons_opt = counter(distance_euclidean)

	dataset = read_corel_images()

	tree_dataset = dataset[:, 1:convert(Int64, floor(size(dataset,2)*0.5))]

	query_dataset = find_points(20000, 100, distance_euclidean, dataset)

	testing_queries, training_queries = make_query_objects(query_dataset, 0)

	weighted_tree, non_weighted_tree = build_trees(tree_dataset, training_queries)

	bson("corel_non_weighted.bson", Dict(:w => weighted_tree, :nw => non_weighted_tree, :tq => testing_queries))
end

function plot_corel_images()

	bsondata = BSON.load("corel.bson")

	weighted_tree = bsondata[:w]

	non_weighted_tree = bsondata[:nw]

	testing_queries = bsondata[:tq]


	dataset = read_corel_images()

	tree_dataset = dataset[:, 1:convert(Int64, floor(size(dataset,2)*0.5))]


	w_results = Vector{Float64}()
	nw_results = Vector{Float64}()
	b_results = Vector{Float64}()

	percent_results = Vector{Float64}()

	io_w = open("plot_latex_rand_w.txt", "w")
	io_nw = open("plot_latex_rand_nw.txt", "w")
	io_b = open("plot_latex_rand_b.txt", "w")

	dataset = read_corel_images()

	size_dataset = convert(Int64, floor(size(dataset,2)*0.5))


	for i=1:25
	#for i=[1,2,3,4,4.2,4.4,4.6,4.8,5]

		for j=1:size(testing_queries,1)
			testing_queries[j].radius = i * 0.1
		end

		w_result, nw_result, b_result, num_result = test_queries(weighted_tree, non_weighted_tree, tree_dataset, distance_euclidean, testing_queries)

		
		println("num result: $num_result")

		w_result_f = 100*w_result/size_dataset
		nw_result_f = 100*nw_result/size_dataset
		b_result_f = 100*b_result/size_dataset

		push!(w_results, w_result)
		push!(nw_results, nw_result)
		push!(b_results, b_result)

		push!(percent_results, num_result/size_dataset)
		y_axis_f = 100*num_result/size_dataset
		write(io_w, "($y_axis_f,$w_result_f)")
		write(io_nw, "($y_axis_f,$nw_result_f)")
		write(io_b, "($y_axis_f,$b_result_f)")

	end

	close(io_nw)
	close(io_w)
	close(io_b)

	println(w_results)
	println(nw_results)
	println(b_results)

	percent_w_results = w_results ./ size_dataset
	percent_nw_results = nw_results ./ size_dataset
	percent_b_results = b_results ./ size_dataset

	savefig(plot(percent_results, [percent_w_results, percent_nw_results, percent_b_results], title="Comparisons",label=["W" "NW" "B"]), "plot_random.png")

end



end
