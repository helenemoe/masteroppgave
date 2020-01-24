using DelimitedFiles
#dist_matrix = readdlm("diffchromall_CharCostFunction2.5.txt")

dist_matrix = [[0 1 2 3 4 5 6]; [1 0 1 2 3 4 5]; [2 1 0 1 2 3 4]; [3 2 1 0 1 2 3]; [4 3 2 1 0 1 2]; [5 4 3 2 1 0 1]; [6 5 4 3 2 1 0]]

#M = maximum(dist_matrix)
#a = 0.4
#Ma = M*a

mutable struct Node
    point        :: Float64
    pot_children :: Vector{Node}
    children :: Vector{Node}
    radius :: Float64
end

function build_ssstree(node)

	if size(node.pot_children,1) < 1
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
				print(dist_matrix[convert(Int64, node.point), convert(Int64, list[i].point)])
				print("\n")
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
#for i = 1:4200
for i = 1:7
	push!(chrom_node, Node(i, Vector{Node}(), Vector{Node}(), 0))

end

print(build_ssstree(Node(0,chrom_node, Vector{Node}(), 0)))
#result = build_ssstree(Node(0,chrom_node, Vector{Node}(), 0))
#print(size(result.children,1))


#print(dist_matrix[1,1])