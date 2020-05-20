module experiment3

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

const NUM_QUERIES = 150

const NUM_TEST_Q = 100

const NUM_TRAIN_Q = 50


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

function setdiff_new(mat1,mat2)
	println(size(mat1))
	mat3 = Array{Float64}(undef, size(mat1,1),0)
	for i=1:size(mat1,2)
		is_in = true
		for j=1:size(mat2,2)
			if mat1[:,i] == mat2[:,j]
				is_in = false
				break
			end
		end
		if is_in
			mat3 = hcat(mat3, mat1[:,i])
		end
	end
	println(size(mat3))
	return mat3
end

function make_query_objects(query_dataset, query_radius)

	test_query_foci_index = sample(1:size(query_dataset,2), convert(Int64,size(query_dataset,2)*(NUM_TEST_Q/NUM_QUERIES)), replace=false)

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
				
				dist = distance_opt(nodey.foci[1], nodex.foci[1])

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


function find_biggest_dist(dataset1, dataset2, modulo, distance_function)

	distance,comparisons = counter(distance_function)

	biggest_dist = 0
	biggest_index = 0

	for i=1:size(dataset1,2)
		if i%modulo != 0
			continue
		end
		temp_dist = 0
		for j=1:size(dataset2,2)
			if j%6 != 0
				continue
			end
			temp_dist += distance(dataset1[:,i], dataset2[:,j])
		end

		if temp_dist > biggest_dist
			biggest_dist = temp_dist
			biggest_index = i
		end
	end
	return biggest_index
end

function build_weighted_tree(dataset, training_queries)
	
	weigthed_base_tree = make_base_tree(dataset)

	weighted_tree = build_multi_ssstree_optimized(weigthed_base_tree, Vector{Float64}(), training_queries)

	return weighted_tree
end

function build_non_weighted_tree(dataset)
	non_weigthed_base_tree = make_base_tree(dataset)

	non_weighted_tree = build_multi_ssstree(non_weigthed_base_tree)

	return non_weighted_tree
end


function build_weighted_bson(dataset, query_dataset, distance_function, filename)

	global distance_opt, comparisons_opt = counter(distance_function)
	
	testing_queries, training_queries = make_query_objects(query_dataset, 0)

	weighted_tree = build_weighted_tree(dataset, training_queries)

	bson(filename, Dict(:w => weighted_tree, :tq => testing_queries))

end

function build_all_weighted_bson(dataset, query_dataset1, query_dataset2, distance_function)

	global distance_opt, comparisons_opt = counter(distance_function)
	
	testing_queries1, training_queries1 = make_query_objects(query_dataset1, 0)
 
	testing_queries2, training_queries2 = make_query_objects(query_dataset2, 0)

	println(size(setdiff(testing_queries1,testing_queries2)))
	println(size(setdiff(training_queries1,training_queries2)))

	weighted_tree_1 = build_weighted_tree(dataset, training_queries1)

	weighted_tree_2 = build_weighted_tree(dataset, training_queries2)

	weighted_tree = build_weighted_tree(dataset, vcat(training_queries1, training_queries2))

	testing_queries = vcat(testing_queries1, testing_queries2)

	println(size(testing_queries))

	bson("o_weighted.bson", Dict(:w => weighted_tree, :tq => testing_queries))

	bson("w1_weighted.bson", Dict(:w1 => weighted_tree_1))

	bson("w2_weighted.bson", Dict(:w2 => weighted_tree_2,))

end



function build_non_weighted_bson(dataset, distance_function)

	global distance,comparisons = counter(distance_function)

	non_weighted_tree = build_non_weighted_tree(dataset)

	bson("non_weighted.bson", Dict(:nw => non_weighted_tree))
end



function make_cluster_queries(dataset, num_clusters, distance_function)

	query_data = find_points(find_biggest_dist(dataset, dataset, 1, distance_function),convert(Int64,floor(NUM_QUERIES/num_clusters)), distance_function, dataset)

	query_before = Set(query_data)
	for i=1:num_clusters-1
		query_data2 = find_points(find_biggest_dist(transpose(setdiff(dataset,query_data)),query_data, 1, distance_function),convert(Int64,floor(NUM_QUERIES/num_clusters)), distance_function, dataset)
		println(size(setdiff(query_data, query_data2)))
		query_data = hcat(query_data, query_data2)
		query_after = Set(query_data)
		@assert query_before != query_after
		query_before = query_after
	end

	return query_data
end

function make_2cluster_queries(dataset, distance_function)

	num_clusters = 2

	println(find_biggest_dist(dataset, dataset, 1, distance_function))

	query_data = find_points(find_biggest_dist(dataset, dataset, 1, distance_function),convert(Int64,floor(NUM_QUERIES/num_clusters)), distance_function, dataset)

	println(find_biggest_dist(setdiff_new(dataset,query_data),query_data, 1, distance_function))

	query_data2 = find_points(find_biggest_dist(setdiff_new(dataset,query_data),query_data, 1, distance_function),convert(Int64,floor(NUM_QUERIES/num_clusters)), distance_function, setdiff_new(dataset, query_data))

	return query_data, query_data2

end

function make_uniform_queries(dataset)

	query_index = sample(1:size(dataset,2), NUM_QUERIES, replace=false)

	query_dataset = dataset[:,query_index[1]]

	for i = 2:NUM_QUERIES
		query_dataset = hcat(query_dataset, dataset[:,query_index[i]])
	end
	return query_dataset
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

	return result, comparisons

end

function test_queries(weighted_tree, weighted_tree_1, weighted_tree_2, non_weighted_tree, dataset, distance_function, testing_queries)

	sum_comparisons_weighted = 0
	sum_comparisons_non_weighted = 0
	sum_comparisons_original = 0

	all_comparisons_weighted = Vector{Float64}()

	all_comparisons_non_weighted = Vector{Float64}()

	all_comparisons_original = Vector{Float64}()

	result_var_bedre = 0

	result_var_bedre_eller_lik = 0

	avg_restult = 0

	test_query_length = size(testing_queries, 1)

	for i = 1:test_query_length

		result_non_weighted, comparisons_non_weighted = search_tree(testing_queries[i], non_weighted_tree, dataset, distance_function)
		result_original, comparisons_original = search_tree(testing_queries[i], weighted_tree, dataset, distance_function)
		result_weighted, comparisons_weighted = search_tree(testing_queries[i], weighted_tree_1, weighted_tree_2, dataset, distance_function)

		sum_comparisons_weighted += comparisons_weighted
		sum_comparisons_non_weighted += comparisons_non_weighted
		sum_comparisons_original += comparisons_original

		avg_restult  += size(result_weighted,1)


		push!(all_comparisons_non_weighted, comparisons_non_weighted)
		push!(all_comparisons_weighted, comparisons_weighted)
		push!(all_comparisons_original, comparisons_original)

		if comparisons_non_weighted > comparisons_weighted
			result_var_bedre += 1
		end

		if comparisons_non_weighted >= comparisons_weighted
			result_var_bedre_eller_lik += 1
		end

	end

	println(sum_comparisons_weighted/test_query_length)
	println(sum_comparisons_non_weighted/test_query_length)
	println(sum_comparisons_original/test_query_length)

	println("result:")
	println(avg_restult/test_query_length)

	println(result_var_bedre/test_query_length)
	println(result_var_bedre_eller_lik/test_query_length)

	return sum_comparisons_weighted/test_query_length, sum_comparisons_non_weighted/test_query_length, sum_comparisons_original/test_query_length, avg_restult/test_query_length
end

function plot_main(dataset, distance_function, radii)

	o_bsondata = BSON.load("o_weighted.bson")
	w1_bsondata = BSON.load("w1_weighted.bson")
	w2_bsondata = BSON.load("w2_weighted.bson")

	bsondata_non_weighted = BSON.load("non_weighted.bson")

	weighted_tree = o_bsondata[:w]

	weighted_tree_1 = w1_bsondata[:w1]

	weighted_tree_2 = w2_bsondata[:w2]

	non_weighted_tree = bsondata_non_weighted[:nw]

	testing_queries = o_bsondata[:tq]

	println("testing : $(size(testing_queries))")

	w_results = Vector{Float64}()
	nw_results = Vector{Float64}()
	o_results = Vector{Float64}()

	num_results = Vector{Float64}()

	io_w = open("plot_latex_w.txt", "w")
	io_nw = open("plot_latex_nw.txt", "w")
	io_o = open("plot_latex_o.txt","w")

	for i=radii

		for j=1:size(testing_queries,1)
			testing_queries[j].radius = i 
		end

		w_comparisons, nw_comparisons, o_comparisons, num_result = test_queries(weighted_tree, weighted_tree_1, weighted_tree_2, non_weighted_tree, dataset, distance_function, testing_queries)

		
		println("num result: $num_result")

		push!(w_results, w_comparisons)
		push!(nw_results, nw_comparisons)
		push!(o_results, o_comparisons)

		push!(num_results, num_result)
		write(io_w, "($num_result,$w_comparisons)")
		write(io_nw, "($num_result,$nw_comparisons)")
		write(io_o, "($num_result,$o_comparisons)")

	end

	close(io_nw)
	close(io_w)
	close(io_o)

	println(w_results)
	println(nw_results)
	println(o_results)

	#savefig(plot(num_results, [w_results, nw_results, b_results], title="Comparisons",label=["W" "NW" "B"]), "plot.png")
	#plot(num_results, w_results)
	#savefig(plot(num_results, [w_results, nw_results, b_results]), "plot.png")
end

#----------------------- Chromosome data set ------------------------------------------------

function build_nw_dist_matrix()
	build_non_weighted_bson(transpose(1:size(dist_matrix,1)), distance_from_matrix)
end

function build_w_facet_dist_matrix()
	dataset = transpose(1:size(dist_matrix,1))
	make_cluster_queries(dataset, 2, distance_from_matrix)
	query_data1, query_data2 = make_2cluster_queries(dataset, distance_from_matrix)
	println(Set(query_data1) == Set(query_data2))
 	build_all_weighted_bson(dataset, query_data1, query_data2, distance_from_matrix)
end

function build_w_dist_matrix(num_clusters)
	dataset = transpose(1:size(dist_matrix,1))
	build_weighted_bson(dataset, make_cluster_queries(dataset, num_clusters, distance_from_matrix), distance_from_matrix)
end

function build_w_dist_matrix_uniform()
	dataset = transpose(1:size(dist_matrix,1))
	build_weighted_bson(dataset, make_uniform_queries(dataset), distance_from_matrix)
end

function dist_matrix_main()
	radii = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
	#radii = [1,20,25,30,35,37,38,40,41,42,43,43.5,44,44.5,45,46,47,48]
	plot_main(transpose(1:size(dist_matrix,1)), distance_from_matrix, radii)
end

#----------------------- Corel data set ------------------------------------------------

function read_corel_images()
	color_moments = readdlm("ColorMoments.asc")
	color_moments = color_moments[:,2:size(color_moments,2)]
	return transpose(color_moments)
end

function build_nw_corel()
	dataset = read_corel_images()
	build_non_weighted_bson(dataset, distance_euclidean)
end

function build_w_facet_corel()
	dataset = read_corel_images()
	query_data1, query_data2 = make_2cluster_queries(dataset, distance_euclidean)
	println(Set(query_data1) == Set(query_data2))
 	build_all_weighted_bson(dataset, query_data1, query_data2, distance_euclidean)
end

function corel_main()
	dataset = read_corel_images()
	radii = [0,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4]
	plot_main(dataset, distance_euclidean, radii)
end


#----------------------- Radnom vectors-------------------------------------------------------

function generate_random_vectors_bson(dimension, tree_size)

	dataset = rand(dimension)
	for i=1:tree_size-1
		dataset = hcat(dataset, rand(dimension))
	end
	bson("random_vectors_dataset.bson", Dict(:d => dataset))
end 

function build_nw_random_vectors()
	dataset = BSON.load("random_vectors_dataset.bson")[:d]
	build_non_weighted_bson(dataset, distance_euclidean)
end

function build_w_facet_random_vectors()
	dataset = BSON.load("random_vectors_dataset.bson")[:d]
	query_data1, query_data2 = make_2cluster_queries(dataset, distance_euclidean)
	println(Set(query_data1) == Set(query_data2))
 	build_all_weighted_bson(dataset, query_data1, query_data2, distance_euclidean)
end

function random_vectors_main()
	dataset = BSON.load("random_vectors_dataset.bson")[:d]
	#radii = [0,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.175,0.18,0.185, 0.19]#1:10 .* 0.01
	#radii = [0,0.1,0.2,0.3,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52]
	radii = [0, 0.5, 0.6, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8]
	#radii = [0.8]
	#radii =  radii .* 0.1
	plot_main(dataset, distance_euclidean, radii)
end

end


