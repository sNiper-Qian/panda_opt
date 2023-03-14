using Dojo
using DojoEnvironments
using IterativeLQR
using LinearAlgebra
using Quaternions

# env = get_environment(:panda, 
#     representation=:minimal, 
#     timestep=0.002,
#     gravity=-0*9.81);
#open(env.vis)

path = joinpath("deps/panda_end_effector.urdf")
dt = 0.002
mech = Mechanism(path; floating=false, gravity=-1*9.81, timestep=dt, keep_fixed_joints=false)
# mech = env.mechanism
joint1 = get_joint(mech,:joint1)
joint2 = get_joint(mech,:joint2)
joint3 = get_joint(mech,:joint3)
joint4 = get_joint(mech,:joint4)
joint5 = get_joint(mech,:joint5)
joint6 = get_joint(mech,:joint6)
joint7 = get_joint(mech,:joint7)
joint8 = get_joint(mech,:jointf1)
joint9 = get_joint(mech,:jointf2)

joints = [joint1;joint2;joint3;joint4;joint5;joint6;joint7;joint8;joint9]

function read_pos_and_vel()
    f1 = open("data/pos.txt", "r")
    f2 = open("data/vel.txt", "r")
    while !eof(f1)
        for i = 1:9
            s = readline(f1)
            if i != 1
                append!(qpos, -parse(Float64, s))
            else
                append!(qpos, parse(Float64, s))
            end
        end
    end
    close(f1)
    while !eof(f2)
        for i = 1:9
            s = readline(f2)
            if i != 1
                append!(qvel, -parse(Float64, s))
            else
                append!(qvel, parse(Float64, s))
            end
        end
    end
    close(f2)
end

function read_ctrl()
    f = open("data/ctrl.txt", "r")
    while !eof(f)
        s = readline(f)
        append!(ctrls, parse(Float64, s))
    end
    close(f)
end

function objective(xs, us, xT, Q, R, p)
    cost = 0
    for i = 1:T-1
        cost += transpose(xs[i]-x_initial[i]) * Q * (xs[i]-x_initial[i])
        # cost += transpose(us[i]) * R * us[i]
        if i < T-1
            cost += transpose((us[i+1]-us[i])/dt) * R * ((us[i+1]-us[i])/dt)
        end
    end
    cost += p * transpose(xs[T]-x_initial[T]) * Q * (xs[T]-x_initial[T])
    return cost
end

function initialize()
    for j = 1:9
        set_minimal_coordinates!(mech, joints[j], [qpos[myfirst*9+j]])
    end
    for j = 1:9
        set_minimal_velocities!(mech, joints[j], [0])
    end
end

# Read qpos and qvel
qpos = zeros(0)
qvel = zeros(0)
read_pos_and_vel()
myfirst = 0
last = Int(length(qpos)/9-1)
T = 440
n = 18
m = 9

# Read contol inputs
ctrls = zeros(0)
read_ctrl()
# us = [zeros(9) for t = 1:T-1]
us = [ctrls[(t-1)*9+1:(t-1)*9+9] for t = 1:T-1]

# Test initial guess
function controller_initial!(mechanism, t)
    for i = 1:3
        gripper_pos_original[i][t] = mech.bodies[1].state.x1[i]
        gripper_vel_original[i][t] = mech.bodies[1].state.v15[i]
    end
    if t == T
        for i = 1:7
            joints_pos_original[i][t] = minimal_coordinates(mech, joints[i])[1]
            joints_vel_original[i][t] = minimal_velocities(mech, joints[i])[1]
        end
        for i = 1:9
            set_input!(joints[i], [0])
        end
    else
        for i = 1:7
            joints_pos_original[i][t] = minimal_coordinates(mech, joints[i])[1]
            joints_vel_original[i][t] = minimal_velocities(mech, joints[i])[1]
        end
        for i = 1:9
            set_input!(joints[i], [us[t][i]])
        end
    end
end
initialize()
gripper_pos = [zeros(T) for _ in 1:3]
gripper_vel = [zeros(T) for _ in 1:3]
gripper_pos_original = [zeros(T) for _ in 1:3]
gripper_vel_original = [zeros(T) for _ in 1:3]
joints_pos_original = [zeros(T) for _ in 1:7]
joints_vel_original = [zeros(T) for _ in 1:7]
storage_initial = simulate!(mech, T*0.002, controller_initial!, record = true)
visualize(mech, storage_initial);

# Initial state
x1 = zeros(18)
for i = 1:7
    x1[(i-1)*2+1] = qpos[myfirst*9+i]
end

# Terminal state
xT = zeros(18)
for i = 1:7
    xT[(i-1)*2+1] = qpos[(myfirst+T-1)*9+i]
end

# IterativeLQR
alpha = 1
alpha_factor = 0.5
lambda = 1
lambda_factor = 10
threshold = 1e-5
lambda_max = 1e5
p = 1
itr = 1
need_rollout = true
# Initialization
z = minimal_to_maximal(mech, x1)
x = x1
u = us[1]
t = [1, 1e-13, 1, 1e-13, 1, 1e-13, 1, 1e-13, 1, 1e-13, 1, 1e-13, 1, 1e-13, 1, 1e-13, 1, 1e-13];
Q = Diagonal(1 * t)
R = Diagonal(1000 * ones(m))
xs = Vector{Float64}[]
for i = 1:T-1
    global u, x, z
    u = us[i]
    push!(xs, x)
    z = step!(mech, z, u) # next x
    x = maximal_to_minimal(mech, z)
    if i == T-1
        push!(xs, x)
    end
end

x_initial = xs
u_initial = us
print("Initial cost:", objective(xs, us, xT, Q, R, p), "\n")

while true
    global z, x, Vx, Vxx, lambda, lambda_factor, lambda_max, alpha, alpha_factor, 
        threshold, itr, need_rollout, us, xs, fxs, fus, lxs, lus, lxxs, luus, luxs, u, xs_new, ks, Ks
    if need_rollout == true
        # Forward rollout
        fxs = Matrix{Float64}[]
        fus = Matrix{Float64}[]
        lxs = Vector{Float64}[]
        lus = Vector{Float64}[]
        lxxs = Matrix{Float64}[]
        luus = Matrix{Float64}[]
        luxs = Matrix{Float64}[]
        x = xs[1]
        z = minimal_to_maximal(mech, x)
        for i = 1:T-1
            global z, u, x, xT, Q, R
            u = us[i]
            fx, fu = get_minimal_gradients!(mech, z, u)
            lx = Q * (x - x_initial[i])
            if i < T-1
                lu = R * (us[i+1] - u) /(dt*dt)
            else
                lu = zeros(9)
            end
            lxx = Q
            luu = -R/(dt*dt)
            lux = zeros(m, n) # 9x18
            push!(fxs, fx)
            push!(fus, fu)
            push!(lxs, lx)
            push!(lus, lu)
            push!(lxxs, lxx)
            push!(luus, luu)
            push!(luxs, lux)
            z = step!(mech, z, u) # next x
            x = maximal_to_minimal(mech, z)
        end
    end

    # Backward pass
    ks = Vector{Float64}[]
    Ks = Matrix{Float64}[]
    # V = transpose(xs[T]-xT) * Diagonal(1.0 * t) * (xs[T]-xT) * 100 
    Vx = Q * (xs[T] - xT) * p # p is penalty factor
    Vxx = Q * p
    for i = T-1:-1:1
        global Vx, Vxx
        Qx = lxs[i] + transpose(fxs[i]) * Vx
        Qu = lus[i] + transpose(fus[i]) * Vx
        Qxx = lxxs[i] + transpose(fxs[i]) * Vxx * fxs[i]
        Qux = luxs[i] + transpose(fus[i]) * Vxx * fxs[i]
        Quu = luus[i] + transpose(fus[i]) * Vxx * fus[i]
        # Regularization
        # eigval = eigvals(Quu)
        # eigvec = eigvecs(Quu)
        # for val in eigval 
        #     if val < 0
        #         val = 0
        #     end
        # end
        # eigval += lambda*ones(9)
        # Quu_inv = eigvec * Diagonal(1.0\eigval) * transpose(eigvec)

        tmp = 0
        while isposdef(Quu) == false
            tmp += 1
            # print(Quu, '\n')
            Quu += lambda * Diagonal(ones(m))
            # print("Regularization...\n")
            if tmp > 10
                break
            end
        end
        Quu_inv = inv(Quu)
        k = -Quu_inv * Qu
        K = -Quu_inv * Qux
        push!(Ks, K)
        push!(ks, k)
        Vx = Qx - transpose(K) * Quu * k
        Vxx = Qxx - transpose(K) * Quu * K
    end

    # Update control input
    x = xs[1]
    z = minimal_to_maximal(mech, x)
    us_new = Vector{Float64}[]
    xs_new = Vector{Float64}[]
    for i = 1:T-1
        global x, z
        u = us[i] + alpha * ks[i] + Ks[i] * (x - xs[i])
        push!(us_new, u)
        push!(xs_new, x)
        z = step!(mech, z, u) 
        x = maximal_to_minimal(mech, z)
        if i == T-1
            push!(xs_new, x)
        end
    end

    cost_old = objective(xs, us, xT, Q, R, p)
    cost_new = objective(xs_new, us_new, xT, Q, R, p)
    if norm(us_new - us) < threshold
        print("Converged\n")
        print("Cost:", cost_old, "\n")
        break
    else
        if cost_new < cost_old
            print("Iteration:", itr, " Cost:", cost_new, "\n")
            itr += 1
            lambda /= lambda_factor
            xs = xs_new
            us = us_new
            need_rollout = true

        else
            print("Iteration:", itr, " Cost:", cost_old, " ", cost_new, "\n")
            itr += 1
            lambda *= lambda_factor
            alpha *= alpha_factor
            need_rollout = false
            if lambda > lambda_max
                print("Lambda over limitation\n")
                break
            end
        end
    end
end

# Test 
function controller!(mechanism, t)
    for i = 1:3
        gripper_pos[i][t] = mech.bodies[1].state.x1[i]
        gripper_vel[i][t] = mech.bodies[1].state.v15[i]
    end
    if t == T
        for i = 1:7
            joints_pos[i][t] = minimal_coordinates(mech, joints[i])[1]
            joints_vel[i][t] = minimal_velocities(mech, joints[i])[1]
            controls[i][t] = 0
            set_input!(joints[i], [0])
        end
    else
        for i = 1:7
            joints_pos[i][t] = minimal_coordinates(mech, joints[i])[1]
            joints_vel[i][t] = minimal_velocities(mech, joints[i])[1]
        end
        for i = 1:9
            # controls[i][t] = u_initial[t][i]
            set_input!(joints[i], [us[t][i]])
        end
    end
end
initialize()
t = range(1, T)
t *= 0.002
controls = [zeros(T) for _ in 1:7]
joints_pos = [zeros(T) for _ in 1:7]
joints_vel = [zeros(T) for _ in 1:7]
joints_pos_set = [zeros(T) for _ in 1:7]
joints_vel_set = [zeros(T) for _ in 1:7]

storage_test = simulate!(mech, T*0.002, controller!, record = true)
visualize(mech, storage_test);
# for i = 1:T
#     for j = 1:7
#         joints_pos_original[j][i] = qpos[(myfirst+i-1)*9+j]
#         joints_vel_original[j][i] = qvel[(myfirst+i-1)*9+j]
#     end
# end
# for i = 1:3
#     gripper_pos_part[i] = gripper_pos_original[i][1:T]
# end
p = plot(t, gripper_pos, label = ["y1" "y2" "y3"], title="Position of end_effector")
p = plot!(t, gripper_pos_original, label = ["o1" "o2" "o3"], ls=:dot)
savefig(p, "plots/ilqr_pos_gripper.png")
p = plot(t, gripper_vel)
p = plot(t, gripper_vel_original)
savefig(p, "plots/ilqr_vel_gripper.png")
p = plot(t, joints_pos, label = ["θ1" "θ2" "θ3" "θ4" "θ5" "θ6" "θ7"], xlabel = "Time [s]", ylabel = "Joint Position [rad]")
p = plot!(t, joints_pos_original, label = ["θ1_b" "θ2_b" "θ3_b" "θ4_b" "θ5_b" "θ6_b" "θ7_b"], ls=:dot)
savefig(p, "plots/ilqr_pos.png")
p = plot(t, joints_vel, label = ["y1" "y2" "y3" "y4" "y5" "y6" "y7"], title="Velocity of joints")
p = plot!(t, joints_vel_original, label = ["o1" "o2" "o3" "o4" "o5" "o6" "o7"], ls=:dot)
savefig(p, "plots/ilqr_vel.png")
p = plot(t, controls, title="Control inputs")
savefig(p, "plots/ilqr_ctrls.png")