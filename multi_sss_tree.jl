module multi_sss_tree

export result

using DelimitedFiles

using DataStructures

using Random

using JuMP, Clp

Random.seed!(0)

dist_matrix = readdlm("diffchromall_CharCostFunction2.5.txt")

N = 4200

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





TEST_QUERY_LENGTH = 100



test_query_foci = rand(1:N, TEST_QUERY_LENGTH)

test_query_radi = rand(0:30, TEST_QUERY_LENGTH)

test_queries = Vector{Query}()

for i = 1:TEST_QUERY_LENGTH
	push!(test_queries, Query(test_query_foci[i], test_query_radi[i]))
end


function build_multi_ssstree(node)

	if size(node.children,1) < 50
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
		print(node.id)
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
			for y = 1:size(children_list,1)
				nodey = children_list[y]
				
				dist = distance(nodey.foci[1], nodex.foci[1])

				if dist < Ma

					push!(children_list[y].children, nodex)
					add_new = 0
					break
				end
			end
			if  add_new == 1
				new_node = nodex
				push!( children_list, new_node )
			end
			
		end

		foci = zeros(0)
		for i = 1: size(children_list,1)
			push!(foci, children_list[i].foci[1])
		end

		X = zeros(Float64, size(foci,1), size(node.children,1))

		for i=1:size(foci,1)
			for j=1:size(node.children, 1)
				X[i,j] = distance(foci[i], node.children[j].id)	
			end
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

multi_sss_children = Vector{MultiFocalNode}()
for i = 1:N/2
#for i = 1:7
	push!(multi_sss_children, MultiFocalNode(i,[i], Vector{MultiFocalNode}(), 0, zeros(0)))

end

multi_sss_test_tree = MultiFocalNode(0,[0],multi_sss_children, 0, zeros(0))

#test_tree = build_multi_ssstree(multi_sss_test_tree)


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

#x = [0 1 2 3 4 5 6; 1 0 1 2 3 4 5; 2 1 0 1 2 3 4; 3 2 1 0 1 2 3; 4 3 2 1 0 1 2; 5 4 3 2 1 0 1; 6 5 4 3 2 1 0]

#z = [0 1 2 3 4 5 6]

#find_weights_LP(x, z)



function build_multi_ssstree_optimized(node, z)

	if size(node.children,1) < 50
	#if size(node.children,1) < 1

		X = zeros(Float64, size(node.foci,1), size(node.children,1))

		for i=1:size(node.foci,1)
			for j=1:size(node.children, 1)
				X[i,j] = distance(node.foci[i], node.children[j].id)	
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
			node.weights = [1]
		"""else
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
			"""
		end

		children_list = Vector{MultiFocalNode}()
		first_node = list[1]
		push!( children_list, first_node )

		add_new = 1

		for x = 2:size(list,1)
			nodex = list[x]
			add_new = 1
			for y = 1:size(children_list,1)
				nodey = children_list[y]
				
				dist = distance(nodey.foci[1], nodex.foci[1])

				if dist < Ma

					push!(children_list[y].children, nodex)
					add_new = 0
					break
				end
			end
			if  add_new == 1
				new_node = nodex
				push!( children_list, new_node )
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
					X[i,j] = distance(node.foci[i], node.children[j].id)	
				end
			end
			optimized_weights, r = coeff(X, z)
			node.weights = vec(optimized_weights)
			node.radius = r
		end



		Z = zeros(0)


		for i=1:size(foci,1)
			sum_dist = 0
			for j=1:TEST_QUERY_LENGTH
				sum_dist += distance(foci[i], test_queries[j].focus) - test_queries[j].radius
			end
			avg_dist = sum_dist/TEST_QUERY_LENGTH
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

z_test = Vector{Float64}()

test_tree_opt = build_multi_ssstree_optimized(multi_sss_test_tree, z_test)

#test_tree = build_multi_ssstree(multi_sss_test_tree)

#print_tree(test_tree_opt)

#print_tree(test_tree)

#println(comparisons())

search_distance, search_comparisons = counter(distance_from_matrix)

results = zeros(0)

saved_distances = zeros(0)

queue = Queue{MultiFocalNode}()

enqueue!(queue, test_tree_opt)
count = 0

function find_range(query, queue, result)
	tree = dequeue!(queue)
	point = query.focus
	range = query.radius
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

			end
			global saved_distances = temp_distances
		else
			temp_distances = saved_distances
			for i = 1:size(tree.foci,1)
				weighted_distance += temp_distances[i]*tree.weights[i]
			end
		end

		if tree.foci[size(tree.foci,1)] == tree.id
			global saved_distances = zeros(0)
		end

		dist_to_query = search_distance(tree.id, point) #MÅ endres!!!!!
		global count += 1

		if dist_to_query - range <= 0
			push!(result, tree.id)
		end


		dist_to_point = weighted_distance

		if dist_to_point <= range + tree.radius
			for i = 1:size(tree.children, 1)
				enqueue!(queue, tree.children[i])
			end
		end
	end
	if ! isempty(queue)
		find_range(query, queue, result)
	else
		return result
	end
end


function find_range_linear_search(query)
	linear_search_distance, liner_search_comparisons = counter(distance_from_matrix)

	result = zeros(0)
	for i=1:N/2
		if linear_search_distance(query.focus, i) <= query.radius
			push!(result, i)
		end
	end
	return result, liner_search_comparisons
end

results_linear, comp_lin = find_range_linear_search(test_queries[5])

println(results_linear)
println(comp_lin())





println(find_range(test_queries[5], queue, results))

#println(result)

println(search_comparisons() - count)

end
