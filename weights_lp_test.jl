
using JuMP, GLPK


function find_weights_LP(x, z)

	model = Model(with_optimizer(GLPK.Optimizer))

	number_of_foci = size(z,1)

	println(number_of_foci)

	println(size(x))


	ONES = ones(number_of_foci)

	println(size(ONES))

	@variable(model, u[1,1:number_of_foci] >= 0)
	@variable(model, v[1,1:number_of_foci] >= 0)
	@variable(model, r)

	@objective(model, Max, sum(u*z) - sum(v*z) - r)
	@constraint(model, sum(u*ONES) + sum(v*ONES) == 1)
	@constraint(model, u*x - v*x .-r .<= 0)

	print(model)

    JuMP.optimize!(model)

    obj_value = JuMP.objective_value(model)
    u_value = JuMP.value.(u)
    v_value = JuMP.value.(v)
    r_value = JuMP.value(r)

    println("Objective value: ", obj_value)
    println("u = ", u_value)
    println("v = ", v_value)
    println("r = ", r_value)

end	

#x = [1 1 2 3 4 5 6; 1 2 1 2 3 4 5; 2 1 2 1 2 3 4; 3 2 1 2 1 2 3; 4 3 2 1 2 1 2; 5 4 3 2 1 2 1; 6 5 4 3 2 1 4]

#z = [0, 1, 2, 3, 4, 5, 6]

x = [1 10; 10 1]

z = [11, 1]

#find_weights_LP(x, z)
