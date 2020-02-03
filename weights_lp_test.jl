
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

#x = [1 10; 10 1]

#z = [11, 1]

#find_weights_LP(x, z)

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

    return wts, rad

end


"""
function find_weights_LP(x, z)

	#print(z)

	model = Model(with_optimizer(GLPK.Optimizer))

	number_of_foci = size(z,1)

	#println(number_of_foci)

	#println(size(x))


	ONES = ones(number_of_foci)

	#println(size(ONES))

	@variable(model, u[1,1:number_of_foci] >= 0)
	@variable(model, v[1,1:number_of_foci] >= 0)
	@variable(model, r)

	@objective(model, Max, sum(u*z) - sum(v*z) - r)
	@constraint(model, sum(u*ONES) + sum(v*ONES) == 1)
	@constraint(model, u*x - v*x .-r .<= 0)

	#print(model)

    JuMP.optimize!(model)

    obj_value = JuMP.objective_value(model)
    u_value = JuMP.value.(u)
    v_value = JuMP.value.(v)
    r_value = JuMP.value(r)
"""
    """
    a = zeros(0)

    println(size(u_value))

    for i = 1:size(u_value, 2)
    	push!(a, u_value[1,i]- v_value[1,i])
    end
    

    return a, r_value

end	
"""


