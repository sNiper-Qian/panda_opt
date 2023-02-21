using Dojo
using DojoEnvironments
using IterativeLQR
using LinearAlgebra
using FiniteDiff
using ReferenceFrameRotations
using Quaternions
using Plots
using SavitzkyGolay

# env = get_environment(:panda, 
#     representation=:minimal, 
#     timestep=0.002,
#     gravity=-0*9.81);

path = joinpath("deps/panda_end_effector.urdf")
dt = 0.002
mech = Mechanism(path; floating=false, gravity=-1*9.81, timestep=dt, keep_fixed_joints=false)
# mech = env.mechanism
# open(env.vis)

# mech = DojoEnvironments.get_mechanism(:panda,
#     timestep=0.002,
#     gravity=-1*9.81,
#     spring=0,
#     damper=10,
#     contact=true,
#     limits=true,
#     )

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

function read_obj_pos(bias)
    f= open("data/obj.txt", "r")
    obj_pos = []
    while !eof(f)
        s = readline(f)
        append!(obj_pos, parse(Float64, s))
    end
    close(f)
    obj_pos -= bias
    return obj_pos
end

function get_final_pos()
    for j = 1:9
        set_minimal_coordinates!(mech, joints[j], [qpos[mylast*9+j]])
    end
    q1 = ReferenceFrameRotations.Quaternion(mech.bodies[1].state.q1.s, mech.bodies[1].state.q1.v1, mech.bodies[1].state.q1.v2, mech.bodies[1].state.q1.v3)
    q1 = quat_to_angle(q1, :XYZ)
    q1 = [q1.a1; q1.a2; q1.a3]
    return mech.bodies[1].state.x1, q1
end

function fill_storage(storage)
    for i = myfirst:mylast
        for j = 1:7
            joints_pos[j][i-myfirst+1] = qpos[i*9+j]
            joints_vel[j][i-myfirst+1] = qvel[i*9+j]
            set_minimal_coordinates!(mech, joints[j], [qpos[i*9+j]])
        end
        gripper_pos_mujoco[1][i-myfirst+1] = norm(mech.bodies[1].state.x1)
        gripper_vel[1][i-myfirst+1] = norm(mech.bodies[1].state.v15)
        for j = 1:9
            storage.x[j][i-myfirst+1] = mech.bodies[j].state.x1
            storage.q[j][i-myfirst+1] = mech.bodies[j].state.q1
        end
    end
end

function initialize()
    for j = 1:7
        set_minimal_coordinates!(mech, joints[j], [qpos[myfirst*9+j]])
    end
    for j = 1:9
        set_minimal_velocities!(mech, joints[j], [0])
    end
    # println([0.38741525; 0.03209606; 1.11761063]-mech.bodies[1].state.x1)   
end

function smooth(pos_start, pos_end, vel_start, s, steps, need_stop, ns)
    v_a = zeros(ns)
    pos_smooth = [zeros(ns) for _ = 1:steps]
    vel_smooth = [zeros(ns) for _ = 1:steps]
    if need_stop == true
        dv = 2 * (pos_end - pos_start - vel_start*(steps-1/2*s)*dt) / ((2*steps-2s)*dt)
        v_a = vel_start + dv
    else
        dv = 2 * (pos_end - pos_start - vel_start*steps*dt) / ((2*steps-s)*dt)
        v_a = vel_start + dv
    end
    pos_smooth[1][1:ns] = pos_start
    vel_smooth[1][1:ns] = vel_start
    for i = 2:s
        for j = 1:ns
            vel_smooth[i][j] = vel_start[j] + (i-1) * dv[j] / s
            # println(qvel_smooth[(i-1)*9+j])
            pos_smooth[i][j] = pos_smooth[i-1][j] + vel_smooth[i][j]*dt
            # println((i-1)*9+j, " ", qpos_smooth[(i-1)*9+j])
        end
    end
    if need_stop == true
        for i = s+1:steps-s
            for j = 1:ns
                vel_smooth[i][j] = v_a[j]
                # println(qvel_smooth[(i-1)*9+j])
                pos_smooth[i][j] = pos_smooth[i-1][j] + vel_smooth[i][j]*dt
                # println((i-1)*9+j, " ", qpos_smooth[(i-1)*9+j])
            end
        end
        for i = steps-s+1:steps
            for j = 1:ns
                vel_smooth[i][j] = vel_smooth[i-1][j] - v_a[j] / s
                # println(qvel_smooth[(i-1)*9+j])
                pos_smooth[i][j] = pos_smooth[i-1][j] + vel_smooth[i][j]*dt
                # println((i-1)*9+j, " ", qpos_smooth[(i-1)*9+j])
            end
        end
    else
        for i = s+1:steps
            for j = 1:ns
                vel_smooth[i][j] = v_a[j]
                # println(qvel_smooth[(i-1)*9+j])
                pos_smooth[i][j] = pos_smooth[i-1][j] + vel_smooth[i][j]*dt
                # println((i-1)*9+j, " ", qpos_smooth[(i-1)*9+j])
            end
        end
    end
    return pos_smooth, vel_smooth
end

function pid!(joint, pos, vel, err_sum, j)
    if j == 1 || j == 3
        kp = 800
        kd = 12
        ki = 0.1
    elseif j == 2
        kp = 900
        kd = 80
        ki = 0.1
    elseif j == 4
        kp = 1600
        kd = 80
        ki = 0.1
    else
        kp = 200
        kd = 8
        ki = 0.1
    end
    p = (pos-minimal_coordinates(mech, joint)[1])*kp
    d = (vel-minimal_velocities(mech, joint)[1])*kd
    # println("setpoint", vel)
    # println("realpoint", minimal_velocities(mech, joint)[1])
    # println(vel-minimal_velocities(mech, joint)[1])
    err_sum += pos-minimal_coordinates(mech, joint)[1]
    i = err_sum*ki
    # println("p",p)
    # println("d",d)
    # println("i",i)
    ctrl = p + d + i
    return ctrl, err_sum
end

function controller!(mechanism, t)
    gripper_pos_pid[1][t] = norm(mech.bodies[1].state.x1-obj_pos)
    gripper_vel[1][t] = norm(mech.bodies[1].state.v15)
    global cost
    for j = 1:9
        if j == 8 || j == 9
            write(f, string(0.0))
            write(f, '\n')
        else
            joints_pos[j][t] = minimal_coordinates(mech, joints[j])[1]
            joints_vel[j][t] = minimal_velocities(mech, joints[j])[1]
            ctrl, err_sum[j] = pid!(joints[j], qpos[(t+myfirst-1)*9+j], qvel[(t+myfirst-1)*9+j], err_sum[j], j)
            controls[j][t] = ctrl
            set_input!(joints[j], [ctrl])
            ctrls[(t-1)*9+j] = ctrl
            write(f, string(ctrl))
            write(f, '\n')
            cost += ctrl*ctrl
        end
    end
end
##
# Read qpos and qvel
qpos = zeros(0)
qvel = zeros(0)
pos_bias = [0.04735242278708274; -0.006168583379561096; 0.5769615312052236] # mujoco_pos - bias = dojo_pos
read_pos_and_vel()
myfirst = 0
mylast = Int(length(qpos)/9-1)
steps = 440
obj_pos = read_obj_pos(pos_bias)
final_pos, final_ori = get_final_pos()
println("Object Position", obj_pos)

# Visualize the trajectory from mujoco
storage = Storage(steps, 9)
t = range(1, steps)
joints_pos = [zeros(steps) for _ in 1:7]
joints_vel = [zeros(steps) for _ in 1:7]
gripper_pos_mujoco = [zeros(steps)]
# gripper_pos_sg = [zeros(steps)]
gripper_vel = [zeros(steps)]
fill_storage(storage)
visualize(mech, storage);
gripper_pos_sg = savitzky_golay(gripper_pos[1], 41, 2)
p = plot(t, gripper_pos_mujoco)
savefig(p, "plots/mujoco_pos_gripper.png")
# p = plot(t[400:500], [gripper_pos[1][400:500], gripper_pos_sg.y[400:500]])
# savefig(p, "plots/mujoco_pos_gripper_sg.png")
p = plot(t, gripper_vel)
savefig(p, "plots/mujoco_vel_gripper.png")
p = plot(t, joints_pos, label = ["θ1" "θ2" "θ3" "θ4" "θ5" "θ6" "θ7"], xlabel = "Time Step [-]", ylabel = "Joint Position [rad]")
savefig(p, "plots/mujoco_pos.png")
p = plot(t, joints_vel, label = ["ω1" "ω2" "ω3" "ω4" "ω5" "ω6" "ω7"], xlabel = "Time Step [-]", ylabel = "Joint Angular Velocity [rad/s]")
savefig(p, "plots/mujoco_vel.png")
plot!(t, joints_pos)

# Do PID control and write control inputs into file
ctrls = zeros(steps*9)
f = open("data/ctrl.txt", "w")
initialize()
# error("1")
t = range(1, steps)
controls = [zeros(steps) for _ in 1:7]
joints_pos = [zeros(steps) for _ in 1:7]
joints_vel = [zeros(steps) for _ in 1:7]
gripper_pos_pid = [zeros(steps) for _ in 1:3]
gripper_vel = [zeros(steps) for _ in 1:3]
err_sum = zeros(9)
cost = 0
storage_pid = simulate!(mech, steps*0.002, controller!, record = true)
println("Original Trajectory")
println("Cost:", cost)
max_angle = zeros(7)
min_angle = zeros(7)
for i = 1:7
    max_angle[i] = maximum(joints_pos[i])
    min_angle[i] = minimum(joints_pos[i])
end
println("Max joint angle:", max_angle)
println("Min joint angle:", min_angle)
visualize(mech, storage_pid);
close(f)
p = plot(t, gripper_pos_pid)
savefig(p, "plots/original_pos_gripper.png")
p = plot(t, gripper_vel)
savefig(p, "plots/original_vel_gripper.png")
p = plot(t, joints_pos)
savefig(p, "plots/original_pos.png")
p = plot(t, joints_vel)
savefig(p, "plots/original_vel.png")
plot!(t, joints_pos)
p = plot(t, controls, title="Control inputs")
savefig(p, "plots/original_ctrls.png")
##

# # Visualize the smooth trajectory
# visualize(mech, storage_smooth);

# Read contol inputs
# ctrls = zeros(0)
# read_ctrl()

# function controller_test!(mechanism, t)
#     if t == steps
#         for i = 1:9
#             set_input!(joints[i], [0])
#         end
#     else
#         for i = 1:9
#             set_input!(joints[i], [ctrls[(t-1)*9+i]])
#         end
#     end
# end
# initialize()
# storage_test = simulate!(mech, steps*0.002, controller_test!, record = true)
# visualize(mech, storage_test);


function controller_smooth!(mechanism, t)
    global cost
    for j = 1:3
        gripper_pos[j][t] = mech.bodies[1].state.x1[j]
        gripper_vel[j][t] = mech.bodies[1].state.v15[j]
    end
    for j = 1:9
        if j == 8 || j == 9
            write(f, string(0.0))
            write(f, '\n')
        else
            joints_pos[j][t] = minimal_coordinates(mech, joints[j])[1]
            joints_vel[j][t] = minimal_velocities(mech, joints[j])[1]
            ctrl, err_sum[j] = pid!(joints[j], qpos_smooth[t][j], qvel_smooth[t][j], err_sum[j], j)
            controls[j][t] = ctrl
            set_input!(joints[j], [ctrl])
            # ctrls[(t-1)*9+j] = ctrl
            write(f, string(ctrl))
            write(f, '\n')
            cost += ctrl*ctrl
        end
    end
end
##
# Smooth the trajectory based on joints
steps = 440
smooth_steps = 440
qpos_smooth, qvel_smooth = smooth(qpos[myfirst*9+1:myfirst*9+9], qpos[mylast*9+1:mylast*9+9], zeros(9), 50, smooth_steps, true, 9)
initialize()
t = range(1, steps)
controls = [zeros(steps) for _ in 1:7]
joints_pos = [zeros(steps) for _ in 1:7]
joints_vel = [zeros(steps) for _ in 1:7]
joints_pos_set = [zeros(steps) for _ in 1:7]
joints_vel_set = [zeros(steps) for _ in 1:7]
joints_pos_original = [zeros(steps) for _ in 1:7]
joints_vel_original = [zeros(steps) for _ in 1:7]
gripper_pos = [zeros(steps) for _ in 1:3]
gripper_vel = [zeros(steps) for _ in 1:3]
err_sum = zeros(9)
cost = 0
f = open("data/ctrl_joints.txt", "w")
storage_smooth = simulate!(mech, smooth_steps*dt, controller_smooth!, record = true)
close(f)
println("Joint based optimized Trajectory")
visualize(mech, storage_smooth);
for i = 1:steps
    for j = 1:7
        joints_pos_set[j][i] = qpos_smooth[i][j]
        joints_vel_set[j][i] = qvel_smooth[i][j]
        joints_pos_original[j][i] = qpos[(myfirst+i-1)*9+j]
        joints_vel_original[j][i] = qvel[(myfirst+i-1)*9+j]
    end
end

p = plot(t, gripper_vel)
# p = plot!(t, gripper_pos_set, label = ["s1"])
savefig(p, "plots/joint_smooth_pos_gripper.png")
p = plot(t, gripper_vel)
# p = plot!(t, gripper_pos_set, label = ["s1"])
savefig(p, "plots/joint_smooth_vel_gripper.png")
p = plot(t, joints_pos, label = ["y1" "y2" "y3" "y4" "y5" "y6" "y7"], title="Position of joints")
p = plot!(t, joints_pos_set, label = ["s1" "s2" "s3" "s4" "s5" "s6" "s7"], ls=:dash)
p = plot!(t, joints_pos_original, label = ["o1" "o2" "o3" "o4" "o5" "o6" "o7"], ls=:dot)
# p = plot(t, joints_pos_set[2], label = ["s2"], ls=:dash, title="Position of joint 2")
# p = plot!(t, joints_pos_original[2], label = ["o2"], ls=:dot)
savefig(p, "plots/joint_smooth_pos.png")
p = plot(t, joints_vel, label = ["y1" "y2" "y3" "y4" "y5" "y6" "y7"], title="Velocity of joints")
p = plot!(t, joints_vel_set, label = ["s1" "s2" "s3" "s4" "s5" "s6" "s7"], ls=:dash)
p = plot!(t, joints_vel_original, label = ["o1" "o2" "o3" "o4" "o5" "o6" "o7"], ls=:dot)
# p = plot(t, joints_vel_set[2], label = ["s2"], ls=:dash, title="Velocity of joint 2")
# p = plot!(t, joints_vel_original[2], label = ["o2"], ls=:dot)
savefig(p, "plots/joint_smooth_vel.png")
p = plot(t, controls)
savefig(p, "plots/joint_ctrls.png")
##

##
function kinematics(q)
    for i = 1:9
        set_minimal_coordinates!(mech, joints[i], [q[i]])
    end
    x1 = mech.bodies[1].state.x1
    q1 = ReferenceFrameRotations.Quaternion(mech.bodies[1].state.q1.s, mech.bodies[1].state.q1.v1, mech.bodies[1].state.q1.v2, mech.bodies[1].state.q1.v3)
    q1 = quat_to_angle(q1, :XYZ)
    q1 = [q1.a1; q1.a2; q1.a3]
    # q1 = mech.bodies[1].state.q1
    # q1 = q1/norm(q1)
    # q1 = [q1.s, q1.v1, q1.v2, q1.v3]
    x = vcat(x1, q1)
    return x
end

function Ja2Jg(q)
    T = zeros(7, 7)
    for i = 1:9
        set_minimal_coordinates!(mech, joints[i], [q[i]])
    end
    q1 = mech.bodies[1].state.q1
    q1 = q1/norm(q1)
    eye = Matrix{Float64}(I, 3, 3)
    T[1:3, 1:3] = eye
    T[4:7, 4:7] =  2*[q1.s q1.v1 q1.v2 q1.v3; 
        -q1.v1 q1.s q1.v3 -q1.v2;
        -q1.v2 -q1.v3 q1.s q1.v1;
        -q1.v3 q1.v2 -q1.v1 q1.s]
    return T
end

function inverse_kinematics_jacobian(q)
    # A = pinv(Ja2Jg(q) * FiniteDiff.finite_difference_jacobian(kinematics, q))
    A = pinv(FiniteDiff.finite_difference_jacobian(kinematics, q))
    # A[:, 4] = zeros(9)
    return A
end

##
function get_endeffector_traj(myfirst, mylast, qpos)
    f = open("data/end_effector.txt", "w")
    for i = myfirst:mylast
        x = kinematics(qpos[i*9+1:i*9+9])
        for t in x
            write(f, string(t))
            write(f, '\n')
        end
    end
    close(f)
end

function read_end_effector(steps, buffer)
    efpos = [zeros(6) for _ = 1:steps+buffer]
    f = open("data/end_effector.txt", "r")
    step = 0
    while !eof(f)
        step += 1
        for i = 1:6
            s = readline(f)
            efpos[step+buffer][i] = parse(Float64, s)
        end
    end
    close(f)
    return efpos
end

function choose_inter_points(efpos, num, steps)
    efpoints = zeros(num*6)
    efpoints[1:6] = efpos[1:6]
    efpoints[(num-1)*6+1:(num-1)*6+6] = efpos[end-5:end]
    interval = Int(floor(steps/(num-1))) 
    for i = 2:num-1
        efpoints[(i-1)*6+1:(i-1)*6+6] = efpos[1+(i-1)*interval*6:6+(i-1)*interval*6]
        # print(1+(i-1)*interval)
    end
    return efpoints, interval
end

function get_endeffector_pos_and_vel(efpoints, num, interval, vel_start, steps)
    cur_steps = 0
    efpos_smooth = zeros(0)
    efvel_smooth = zeros(0)
    pos_start = efpoints[1:6] 
    pos_end = efpoints[7:12]
    # Smooth all segments except the mylast one
    for i = 1:num-2
        p, v = smooth(pos_start, pos_end, vel_start, Int(floor(interval*0.1)), interval, false, 6)
        efpos_smooth = [efpos_smooth; p]
        efvel_smooth = [efvel_smooth; v]
        # println(pos_end)
        pos_start = efpos_smooth[end-5:end]
        # println(pos_start)
        pos_end = efpoints[(i+1)*6+1:(i+1)*6+6]
        vel_start = efvel_smooth[end-5:end]
        cur_steps += interval
    end
    p, v = smooth(pos_start, pos_end, vel_start, Int(floor((steps - cur_steps)*0.1)), steps - cur_steps, true, 6)
    efpos_smooth = [efpos_smooth; p]
    efvel_smooth = [efvel_smooth; v]
    return efpos_smooth, efvel_smooth
end

function get_ef_vel(efpos,steps)
    efvel = [zeros(6) for _ = 1:steps]
    # efvel2 = zeros(steps*7)
    for i = 1:steps-1
        x = efpos[i][1:3]
        next_x = efpos[i+1][1:3]
        efvel[i][1:3] = (next_x - x)/dt
        q = efpos[i][4:6]
        next_q = efpos[i+1][4:6]
        efvel[i][4:6] = (next_q - q)/dt
        # q = ReferenceFrameRotations.Quaternion(efpos[(i-1)*7+4:(i-1)*7+7])
        # next_q = ReferenceFrameRotations.Quaternion(efpos[i*7+4:i*7+7])
        # w = 2*inv(q)*(next_q - q)/dt
        # println("Angular velocity:", w)
        # # w = (next_q - q)/dt
        # efvel[(i-1)*7+4:(i-1)*7+7] = [w.q0, w.q1, w.q2, w.q3]
        # efvel2[(i-1)*7+4:(i-1)*7+7] = [0, w.q1, w.q2, w.q3]
        # efvel2[(i-1)*7+4] = sqrt((2/dt)^2 - norm(efvel2[(i-1)*7+5:(i-1)*7+7])^2) - 2/dt
    end
    return efvel
end

function extend_traj(efpos, w, steps)
    half_w = Int((w-1)/2)
    efpos_extend = [zeros(6) for _ = 1:steps+half_w]
    for i = half_w+1:half_w+steps
        efpos_extend[i][:] = efpos[i-half_w]
    end
    for i = half_w:-1:1
        efpos_extend[i][:] = efpos[1][:] - (efpos[half_w-i+2][:] - efpos[1][:])
    end
    # for i = steps + half_w + 1 : steps + half_w*3
    #     efpos_extend[i][:] = efpos[end][:]
    # end
    return efpos_extend
end

function sg_filter(efpos, efpos_smooth, l)
    efpos_mat = mapreduce(permutedims, vcat, efpos);
    efpos_smooth_mat = mapreduce(permutedims, vcat, efpos_smooth);
    for i = 1:6
        efpos_smooth_mat[:,i] = savitzky_golay(efpos_mat[:,i], w, 2).y
    end
    efpos = [efpos_mat[i, :] for i = 1:l]
    efpos_smooth = [efpos_smooth_mat[i, :] for i = 1:l]
    return efpos_smooth
end

function ef2joints(efvel_smooth, qpos_start, qvel_start, steps)
    qpos_from_ef = [zeros(9) for _ = 1:steps]
    qvel_from_ef = [zeros(9) for _ = 1:steps]
    qpos_from_ef[1][1:9] = qpos_start
    qvel_from_ef[1][1:9] = qvel_start
    f1 = open("data/qpos_from_ef.txt", "w")
    f2 = open("data/qvel_from_ef.txt", "w")
    for j = 1:9        
        write(f1, string(qpos_from_ef[1][j]))
        write(f1, '\n')
        write(f2, string(qvel_from_ef[1][j]))
        write(f2, '\n')
    end
    for i = 2:steps
        qvel_from_ef[i][1:9] = inverse_kinematics_jacobian(qpos_from_ef[i-1][1:9]) * efvel_smooth[i][1:6]
        # qvel_jan3[i] = qvel_from_ef[(i-1)*9+1:(i-1)*9+9]
        # println("Inverse of Geometric Jacobian:")
        # display(inverse_kinematics_jacobian(qpos_from_ef[(i-2)*9+1:(i-2)*9+9]))
        # println(efvel_smooth[(i-1)*7+1:(i-1)*7+7])
        # println(qvel_from_ef[(i-1)*9+1:(i-1)*9+9])
        qpos_from_ef[i][1:9] = qpos_from_ef[i-1][1:9] + qvel_from_ef[i][1:9] * dt
        for j = 1:9        
            write(f1, string(qpos_from_ef[i][j]))
            write(f1, '\n')
            write(f2, string(qvel_from_ef[i][j]))
            write(f2, '\n')
        end
    end
    close(f1)
    close(f2)
end

function read_q_from_ef(steps)
    f1 = open("data/qpos_from_ef.txt", "r")
    f2 = open("data/qvel_from_ef.txt", "r")
    qpos_from_ef = [zeros(9) for _ = 1:steps]
    qvel_from_ef = [zeros(9) for _ = 1:steps]
    step = 0
    while !eof(f1)
        step += 1
        for i = 1:9
            s = readline(f1)
            qpos_from_ef[step][i] = parse(Float64, s)
        end
    end
    close(f1)
    step = 0
    while !eof(f2)
        step += 1
        for i = 1:9
            s = readline(f2)
            qvel_from_ef[step][i] = parse(Float64, s)
        end
    end
    close(f2)
    return qpos_from_ef, qvel_from_ef
end

function fill_storage_ef(storage, efpoints)
    for i = 1:steps
        for j = 1:7
            set_minimal_coordinates!(mech, joints[j], [efpoints[(i-1)*9+j]])
        end
        for j = 1:9
            storage.x[j][i] = mech.bodies[j].state.x1
            storage.q[j][i] = mech.bodies[j].state.q1
            # storage.x[j][i] = efpoints[(i-1)*7+1:(i-1)*7+3]
            # q = angle_to_quat(efpoints[(i-1)*6+4], efpoints[(i-1)*6+5], efpoints[(i-1)*6+6], :XYZ)
            # storage.q[j][i] = Quaternions.Quaternion(efpoints[(i-1)*7+4:(i-1)*7+7])
        end
    end
end

function controller_ef_smooth!(mechanism, t)
    global cost
    for j = 1:3
        gripper_pos[j][t] = mech.bodies[1].state.x1[j]
        gripper_vel[j][t] = mech.bodies[1].state.v15[j]
    end
    for j = 1:9
        if j == 8 || j == 9
            write(f, string(0.0))
            write(f, '\n')
        else
            joints_pos[j][t] = minimal_coordinates(mech, joints[j])[1]
            joints_vel[j][t] = minimal_velocities(mech, joints[j])[1]
            # println(joints_pos[j][t])
            # println(joints_vel[j][t])
            ctrl, err_sum[j] = pid!(joints[j], qpos_from_ef[t][j], qvel_from_ef[t][j], err_sum[j], j)
            controls[j][t] = ctrl
            # println(t)
            # println("c", ctrl)
            set_input!(joints[j], [ctrl])
            ctrls[(t-1)*9+j] = ctrl
            cost += ctrl*ctrl
            write(f, string(ctrl))
            write(f, '\n')
        end
    end
end
##
# Smooth the trajectory based on end_effector
steps = 440
cost = 0
w = 101
half_w = Int((w-1)/2)
get_endeffector_traj(myfirst, myfirst+steps-1, qpos)
efpos = read_end_effector(steps, 0)
efpos_extend = extend_traj(efpos, w, steps)
efvel_extend = get_ef_vel(efpos_extend, steps+3*half_w)
efpos_smooth = [zeros(6) for _ = 1:steps+3*half_w]
efpos_smooth = sg_filter(efpos_extend, efpos_smooth, steps+3*half_w)
# num_points = 5
# efpoints, interval = choose_inter_points(efpos, num_points, steps)
# efpos_smooth, efvel_smooth = get_endeffector_pos_and_vel(efpoints, num_points, interval, zeros(6), steps)
efvel_smooth = get_ef_vel(efpos_smooth, steps+3*half_w)
efpos_extend = efpos_extend[half_w+1:end]
efvel_extend = efvel_extend[half_w+1:end]
efpos_smooth = efpos_smooth[half_w+1:end]
efvel_smooth = efvel_smooth[half_w+1:end]
steps += half_w*2
# ef2joints(efvel_smooth, qpos[myfirst*9+1:myfirst*9+9], zeros(9), steps)
qpos_from_ef, qvel_from_ef = read_q_from_ef(steps)
initialize()
f = open("data/ctrl_ef.txt", "w")
t = range(1, steps)
controls = [zeros(steps) for _ in 1:7]
joints_pos = [zeros(steps) for _ in 1:7]
joints_vel = [zeros(steps) for _ in 1:7]
joints_pos_set = [zeros(steps) for _ in 1:7]
joints_vel_set = [zeros(steps) for _ in 1:7]
joints_pos_original = [zeros(steps) for _ in 1:7]
joints_vel_original = [zeros(steps) for _ in 1:7]
gripper_pos = [zeros(steps) for _ in 1:3]
gripper_vel = [zeros(steps) for _ in 1:3]
gripper_pos_set = [zeros(steps) for _ in 1:3]
gripper_vel_set = [zeros(steps) for _ in 1:3]
gripper_pos_original = [zeros(steps) for _ in 1:3]
gripper_vel_original = [zeros(steps) for _ in 1:3]
err_sum = zeros(9)
storage_ef_smooth = Storage(steps, 9)
# fill_storage_ef(storage_ef_smooth, qpos_from_ef)
ctrls = zeros(9*steps)
storage_ef_smooth = simulate!(mech, steps*dt, controller_ef_smooth!, record = true)
# storage_ef_smooth = simulate!(mech, 1:steps, storage_ef_smooth, controller_ef_smooth!, record = true, verbose = false)
close(f)
println("End effector based optimized Trajectory")
# println("Cost", cost)
# for i = 1:7
#     max_angle[i] = maximum(joints_pos[i])
#     min_angle[i] = minimum(joints_pos[i])
# end
# println("Max joint angle:", max_angle)
# println("Min joint angle:", min_angle)
for t = 1:steps
    for i = 1:3
        gripper_pos_original[i][t] = efpos_extend[t][i]
        gripper_pos_set[i][t] = efpos_smooth[t][i]
        gripper_vel_original[i][t] = efvel_extend[t][i]
        gripper_vel_set[i][t] = efvel_smooth[t][i]
    end
    for i = 1:7
        joints_pos_set[i][t] = qpos_from_ef[t][i]
        joints_vel_set[i][t] = qvel_from_ef[t][i]
        if (myfirst+t-1)*9+i <= length(qpos)
            joints_pos_original[i][t] = qpos[(myfirst+t-1)*9+i]
            joints_vel_original[i][t] = qvel[(myfirst+t-1)*9+i]
        else
            joints_pos_original[i][t] = qpos[mylast*9+i]
            joints_vel_original[i][t] = qvel[mylast*9+i]
        end
    end
end
visualize(mech, storage_ef_smooth);
# print(gripper_pos_set[1]-gripper_pos[1])
p = plot(t, gripper_pos, label = ["y1" "y2" "y3"], title="Position of end_effector")
p = plot!(t, gripper_pos_set, label = ["s1" "s2" "s3"], ls=:dash)
p = plot!(t, gripper_pos_original, label = ["o1" "o2" "o3"], ls=:dot)
# p = plot(t, gripper_pos_set[2], label = ["s2"], title="Position of end_effector in y axis")
# p = plot!(t, gripper_pos_original[2], label = ["o2"], ls=:dot)
savefig(p, "plots/ef_smooth_pos_gripper.png")
p = plot(t, gripper_vel, label = ["y1" "y2" "y3"])
p = plot!(t, gripper_vel_set, label = ["s1" "s2" "s3"], ls=:dash)
# p = plot!(t, gripper_pos_original, label = ["o1" "o2" "o3"], ls=:dot)
savefig(p, "plots/ef_smooth_vel_gripper.png")
p = plot(t, joints_pos, label = ["y1" "y2" "y3" "y4" "y5" "y6" "y7"], title="Position of joints")
p = plot!(t, joints_pos_set, label = ["s1" "s2" "s3" "s4" "s5" "s6" "s7"], ls=:dash)
# p = plot!(t, joints_pos_original, label = ["s1" "s2" "s3" "s4" "s5" "s6" "s7"], ls=:dot)
savefig(p, "plots/ef_smooth_pos.png")
p = plot(t, joints_vel, label = ["y1" "y2" "y3" "y4" "y5" "y6" "y7"])
p = plot!(t, joints_vel_set, label = ["s1" "s2" "s3" "s4" "s5" "s6" "s7"], ls=:dash)
savefig(p, "plots/ef_smooth_vel.png")
p = plot(t, controls)
savefig(p, "plots/ef_smooth_ctrls.png")
##

##
function check_near_object(obj_pos, gripper_pos, r)
    if norm(obj_pos-gripper_pos) <= r
        return true
    else
        # println(norm(obj_pos-gripper_pos))
        return false
    end
end

function get_period_one(myfirst, least, mylast, qpos, r)
    for i = myfirst+least:mylast
        for j = 1:7
            set_minimal_coordinates!(mech, joints[j], [qpos[i*9+j]])
        end
        gripper_pos = mech.bodies[1].state.x1
        if check_near_object(obj_pos, gripper_pos, r)
            return i-myfirst
        end
    end
end

function split_traj(qpos, myfirst, mylast, steps)
    r = 0.3
    least = Int(floor(steps*0.5))
    period_one = get_period_one(myfirst, least, mylast, qpos, r)
    println("Period_one:", period_one)
    period_two = steps - period_one
    qpos_1 = qpos[myfirst*9+1:(myfirst+period_one-1)*9+9]
    qpos_2 = qpos[(myfirst+period_one)*9+1:mylast*9+9]
    qpos_smooth, qvel_smooth = smooth(qpos_1[1:9], qpos_1[end-8:end], zeros(9), Int(floor(period_one*0.1)), period_one, true, 9)
    pos_start = kinematics(qpos_smooth[end])
    vel_start = zeros(6)
    qvel_start = zeros(9)
    get_endeffector_traj(0, Int(length(qpos_2)/9-1), qpos_2)
    efpos = read_end_effector(period_two, 0)
    buffer = 40
    # efpos[1] = efpos[buffer+1]
    c = efpos
    # num_points = 5
    # efpoints, interval = choose_inter_points(efpos, num_points, period_two)
    # efpos_smooth, efvel_smooth = get_endeffector_pos_and_vel(efpoints, num_points, interval, vel_start, period_two)
    efpos_extend = extend_traj(efpos, w, period_two)
    # efvel_extend = get_ef_vel(efpos_extend, period_two+half_w)
    efpos_smooth = [zeros(6) for _ = 1:period_two+half_w]
    efpos_smooth = sg_filter(efpos_extend, efpos_smooth, period_two+half_w)
    extend_vector = [efpos_smooth[end] for _ = 1:4*half_w]
    efpos_smooth = [efpos_smooth;extend_vector]
    efvel_smooth = get_ef_vel(efpos_smooth, period_two+5*half_w)
    efpos_extend = efpos_extend[half_w+1:end]
    # efvel_extend = efvel_extend[half_w+1:end]
    efpos_smooth = efpos_smooth[half_w+1:end]
    efvel_smooth = efvel_smooth[half_w+1:end]
    buffer_vector = [zeros(6) for _ = 1:buffer]
    efpos_smooth = [buffer_vector;efpos_smooth]
    efvel_smooth = [buffer_vector;efvel_smooth]
    # v_d = 2*(efpos_smooth[buffer+1]-pos_start)/(buffer*dt)
    v_d = efvel_smooth[buffer+1]
    a_d = v_d/buffer
    v = zeros(6)
    for i = 1:buffer
        v = v + a_d
        efvel_smooth[i] = v
        if i == 1
            efpos_smooth[i] = pos_start + v*dt
        else
            efpos_smooth[i] = efpos_smooth[i-1] + v*dt
        end
    end
    c = efvel_smooth
    # display(efpos_smooth)
    period_two += 4*half_w+buffer
    println("Period_two:", period_two)
    @time ef2joints(efvel_smooth, qpos_2[1:9], qvel_start, period_two)
    qpos_from_ef, qvel_from_ef = read_q_from_ef(period_two)
    a = qpos_from_ef
    b = qvel_from_ef
    qpos_output = [qpos_smooth; qpos_from_ef]
    qvel_output = [qvel_smooth; qvel_from_ef]
    return efvel_smooth, qpos_output, qvel_output, a, b, c
end

function controller_integrated!(mechanism, t)
    global cost
    for j = 1:3
        gripper_pos[j][t] = mech.bodies[1].state.x1[j]-final_pos[j]
        q1 = ReferenceFrameRotations.Quaternion(mech.bodies[1].state.q1.s, mech.bodies[1].state.q1.v1, mech.bodies[1].state.q1.v2, mech.bodies[1].state.q1.v3)
        q1 = quat_to_angle(q1, :XYZ)
        q1 = [q1.a1; q1.a2; q1.a3]
        gripper_ori[j][t] = q1[j]-final_ori[j]
        if j == 3
            gripper_pos[j][t] += 0.04
        end
    end
    for j = 1:9
        if j == 8 || j == 9
            write(f, string(0.0))
            write(f, '\n')
        else
            joints_pos[j][t] = minimal_coordinates(mech, joints[j])[1]
            joints_vel[j][t] = minimal_velocities(mech, joints[j])[1]
            ctrl, err_sum[j] = pid!(joints[j], qpos_integrated[t][j], qvel_integrated[t][j], err_sum[j], j)
            # println(t, ctrl)
            set_input!(joints[j], [ctrl])
            # ctrls[(t-1)*9+j] = ctrl
            cost += ctrl*ctrl
            write(f, string(ctrl))
            write(f, '\n')
        end
    end
end

##
# Integrated Trajectory
steps = 440
cost = 0
w = 101
half_w = Int((w-1)/2)
efvel_smooth, qpos_integrated, qvel_integrated, a, b, c = split_traj(qpos, myfirst, mylast, steps)
buffer = 40
steps += half_w*4+buffer
initialize()
joints_pos = [zeros(steps) for _ in 1:7]
joints_vel = [zeros(steps) for _ in 1:7]
joints_pos_set = [zeros(steps) for _ in 1:7]
joints_vel_set = [zeros(steps) for _ in 1:7]
gripper_pos = [zeros(steps) for _ in 1:3]
gripper_ori = [zeros(steps) for _ in 1:3]
gripper_vel = [zeros(steps) for _ in 1:3]
err_sum = zeros(9)
f = open("data/ctrl_integrated.txt", "w")
storage_integrated = simulate!(mech, steps*dt, controller_integrated!, record = true, verbose = false)
close(f)
println("Integrated Trajectory")
# println("Cost", cost)
# for i = 1:7
#     max_angle[i] = maximum(joints_pos[i])
#     min_angle[i] = minimum(joints_pos[i])
# end
# println("Max joint angle:", max_angle)
# println("Min joint angle:", min_angle)
visualize(mech, storage_integrated);
for i = 1:steps
    for j = 1:7
        joints_pos_set[j][i] = qpos_integrated[i][j]
        joints_vel_set[j][i] = qvel_integrated[i][j]
    end
end
buffer_vector = [zeros(6) for _ = 1:buffer]
t = 1:steps
p = plot(t, gripper_pos, label = ["Δpx" "Δpy" "Δpz"], xlabel = "Time Step [-]", ylabel = "Deviation of position [m]")
savefig(p, "plots/integrated_pos_gripper.png")
for i = 1:6
    gripper_ori[1][i] = 0.5
end
p = plot(t, gripper_ori, label = ["Δθx" "Δθy" "Δθz"], xlabel = "Time Step [-]", ylabel = "Deviation of orientation [rad]")
savefig(p, "plots/integrated_ori_gripper.png")
p = plot(t, gripper_vel)
savefig(p, "plots/integrated_vel_gripper.png")
p = plot(t, joints_pos, label = ["θ1_r" "θ2_r" "θ3_r" "θ4_r" "θ5_r" "θ6_r" "θ7_r"], xlabel = "Time Step [-]", ylabel = "Joint Position [rad]")
p = plot!(t, joints_pos_set, label = ["θ1" "θ2" "θ3" "θ4" "θ5" "θ6" "θ7"], ls=:dash)
savefig(p, "plots/integrated_pos_pid.png")
p = plot(t, joints_vel, label = ["ω1_r" "ω2_r" "ω3_r" "ω4_r" "ω5_r" "ω6_r" "ω7_r"], xlabel = "Time Step [-]", ylabel = "Joint Angular Velocity [rad/s]")
p = plot!(t, joints_vel_set, label = ["ω1" "ω2" "ω3" "ω4" "ω5" "ω6" "ω7"], ls=:dash)
savefig(p, "plots/integrated_vel_pid.png")
##


# # Do the optimization
# n = env.num_states
# m = env.num_inputs
# T = steps

# dyn = IterativeLQR.Dynamics(
#     (y, x, u, w) -> dynamics(y, env, x, u, w), 
#     (dx, x, u, w) -> dynamics_jacobian_state(dx, env, x, u, w),
#     (du, x, u, w) -> dynamics_jacobian_input(du, env, x, u, w),
#     n, n, m)
# model = [dyn for t = 1:T-1]

# # Initial state
# x1 = zeros(18)
# for i = 1:7
#     x1[(i-1)*2+1] = qpos[myfirst*9+i]
# end

# # Terminal state
# xT = zeros(18)
# for i = 1:7
#     xT[(i-1)*2+1] = qpos[mylast*9+i]
# end

# # Ctrl input from pid controller
# u_guess = [ctrls[(t-1)*9+1:(t-1)*9+9] for t = 1:T-1]

# # Trajectory of initial guess
# traj = IterativeLQR.rollout(model, x1, u_guess)
# storage = generate_storage(mech, [env.representation == :minimal ? minimal_to_maximal(mech, x) : x for x in traj])
# visualize(mech, storage, vis=env.vis, build=true)

# # Objective function
# t = [1, 0.001, 1, 0.001, 1, 0.001, 1, 0.001, 1, 0.001, 1, 0.001, 1, 0.001, 0.001, 0.001, 0.001, 0.001]; # Currently dont care about fingers

# function ot(x, u)
#     return transpose(x-xT) * Diagonal(1.0 * t) * (x-xT) + transpose(u) * Diagonal(1.0 * ones(m)) * u
# end

# function oT(x, u)
#     return transpose(x-xT) * Diagonal(1.0 * t) * (x-xT)
# end
# # ot = (x, u, w) -> transpose(x-xT) * Diagonal(1.0 * t) * (x-xT) + transpose(u) * Diagonal(1.0 * ones(m)) * u
# # oT = (x, u, w) -> transpose(x-xT) * Diagonal(1.0 * t) * (x-xT)
# ct = IterativeLQR.Cost(ot, n, m)
# cT = IterativeLQR.Cost(oT, n, 0)
# obj = [[ct for t = 1:T-1]..., cT]

# # constraints
# function initial_constraint(x, u)
#     cons = x - x1
#     return cons
# end

# function terminal_constraint(x, u)
#     cons = x - xT
#     return cons
# end
# con1 = IterativeLQR.Constraint(initial_constraint, n, 0)
# cont = IterativeLQR.Constraint()
# conT = IterativeLQR.Constraint(terminal_constraint, n, 0)
# cons = [con1, [cont for t = 2: T-1]..., conT]

# #Solver
# s = IterativeLQR.Solver(model, obj, cons,
#     options = IterativeLQR.Options(
#         max_iterations = 30, 
#         verbose = true))
# IterativeLQR.initialize_controls!(s, u_guess)
# IterativeLQR.initialize_states!(s, traj)

# #Solve
# # @time IterativeLQR.solve!(s)
# IterativeLQR.solve!(s)

# #Solution
# traj_sol, u_sol = IterativeLQR.get_trajectory(s)
# @show IterativeLQR.eval_obj(s.m_data.obj.costs, s.m_data.x, s.m_data.u, s.m_data.w)
# @show s.s_data.iter[1]
# @show norm(terminal_constraint(s.m_data.x[T], zeros(0)), Inf)

# #Visualize
# storage = generate_storage(mech, [env.representation == :minimal ? minimal_to_maximal(mech, x) : x for x in traj_sol])
# visualize(mech, storage, vis=env.vis, build=true)

# qpos_jan = [zeros(9) for i=1:steps]

# for i=1:steps
#     qpos_jan[i] = qpos[(i-1)*9+1:i*9]
# end

# qvel_jan = [zeros(9) for i=1:steps]

# for i=1:steps
#     qvel_jan[i] = qvel[(i-1)*9+1:i*9]
# end

# evel_jan = [zeros(7) for i=1:steps]

# for i=1:steps
#     evel_jan[i] = kinematics_jacobian_jan(q_jan[i])*qvel_jan[i]
# end

# qvel_jan2 = [zeros(7) for i=1:steps]

# for i=1:steps
#     qvel_jan2[i] = inverse_kinematics_jacobian(q_jan[i])*efvel_smooth[i]
# end