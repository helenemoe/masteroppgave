using DelimitedFiles
dist_matrix = readdlm("diffchromall_CharCostFunction2.5.txt")

#dist_matrix = [[0 1 2 3 4 5 6]; [1 0 1 2 3 4 5]; [2 1 0 1 2 3 4]; [3 2 1 0 1 2 3]; [4 3 2 1 0 1 2]; [5 4 3 2 1 0 1]; [6 5 4 3 2 1 0]]

#M = maximum(dist_matrix)
#a = 0.4
#Ma = M*a

mutable struct Node
    point        :: Float64
    pot_children :: Vector{Node}
    children :: Vector{Node}
    radius :: Float64
end

mutable struct MultiFocalNode
    foci        :: Vector{Float64}
    children :: Vector{MultiFocalNode}
    radius :: Float64
    weights :: Vector{Float64}
end

function build_ssstree(node)

	if size(node.pot_children,1) < 50
		node.children = node.pot_children
		node.pot_children = Vector{Node}()
		node
	else

		list = node.pot_children
		max = 0
		for x = 1:size(list,1) 
			for y = 1:size(list,1) 
				if dist_matrix[convert(Int64,list[x].point),convert(Int64,list[y].point)]>max
					max = dist_matrix[convert(Int64,list[x].point),convert(Int64,list[y].point)]
				end
			end
		end
		Ma = 0.4*max
		radius = 0
		if node.point == 0
			node.radius = 100000
		else
			for i = 1:size(list,1)
				if dist_matrix[convert(Int64, node.point), convert(Int64, list[i].point)] > radius
					radius = dist_matrix[convert(Int64, node.point), convert(Int64, list[i].point)]
				end
			end
			node.radius = radius
		end

		children_list = Vector{Node}()
		first_node = list[1]
		push!( children_list, first_node )

		add_new = 1

		for x = 2:size(list,1)
			nodex = list[x]
			add_new = 1
			for y = 1:size(children_list,1)
				nodey = children_list[y]
				
				dist = dist_matrix[convert(Int64, nodey.point), convert(Int64, nodex.point)]

				if dist < Ma

					push!(children_list[y].pot_children, nodex)
					add_new = 0
					break
				end
			end
			if  add_new == 1
				new_node = nodex
				push!( children_list, new_node )
			end
			
		end

		for i = 1:size(children_list,1)
			push!(node.children, build_ssstree(children_list[i]))
		end


		node.pot_children = Vector{Node}()

		node

	end
end


chrom_node = Vector{Node}()
for i = 1:4200
#for i = 1:7
	push!(chrom_node, Node(i, Vector{Node}(), Vector{Node}(), 0))

end

tree = build_ssstree(Node(0,chrom_node, Vector{Node}(), 0))
#result = build_ssstree(Node(0,chrom_node, Vector{Node}(), 0))
#print(size(result.children,1))

function find_node(point, tree)
	if tree.point == point
		return 1
	else
		if size(dist_matrix,1) < point
			return 0
		end
		for i = 1:size(tree.children, 1)
			dist_to_point = dist_matrix[convert(Int64, tree.children[i].point), convert(Int64, point)]

			if dist_to_point <= tree.children[i].radius
				result = find_node(point, tree.children[i])

				if result == 1
					return result
				end
			end
		end
		return 0
	end
end


print(find_node(6, tree))

result = zeros(0)
comparisons = 0

function find_range(point, range, tree)
	if tree.point == 0.0
		for i = 1:size(tree.children, 1)
			find_range(point, range, tree.children[i])
		end
	else

		dist_to_point = dist_matrix[convert(Int64, tree.point), convert(Int64, point)]
		if dist_to_point - range <= 0
			global comparisons = comparisons + 1
			push!(result, tree.point)
		end
		if dist_to_point < range + tree.radius
			global comparisons = comparisons + 1
			for i = 1:size(tree.children, 1)
				find_range(point, range, tree.children[i])
			end
		end
	end
end

find_range(1,30, tree)

print(result)

print("antall sammenligninger \n")
print(comparisons)

multi_focal_tree = MultiFocalNode(zeros(0), Vector{MultiFocalNode}(), 0, zeros(0))
function make_ambit_tree(tree)
	foci = zeros(0)
	for i = 1: size(tree.children,1)
		push!(foci, tree.children[i].point)
	end
	for i = 1: size(tree.children,1)
		weights = zeros(0)
		for j = 1:size(foci,1)
			if j == i
				push!(weights,1)
			else
				push!(weights,0)
			end
		end
		mfn = MultiFocalNode(foci, Vector{MultiFocalNode}(),  tree.children[i].radius, weights)
		push!(multi_focal_tree.children, mfn)
	end
end

make_ambit_tree(tree)
print(multi_focal_tree)

	

