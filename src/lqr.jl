using LinearAlgebra
using Dojo
using ControlSystemsBase
using DojoEnvironments

dt = 0.002

path = joinpath("deps/panda_end_effector.urdf")
# path = joinpath("panda_lqr/deps/panda_no_end_effector.urdf")

mech = Mechanism(path; floating=false, gravity=-0*9.81, timestep=dt, keep_fixed_joints=false)

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
# joints = [joint1;joint2;joint3;joint4;joint5;joint6]

qpos = [0.011428426107260648;
-0.3850697375438758;
0.11806686562359735;
-1.982577344796074;
0.01612098241914294;
1.5772243399948567;
-0.6755980910623021;
6.802225047151906e-05;
0.0006212611515841422]
# qpos = [0;-0.5;0;1.0;0;-1.5]

nq = length(qpos)

for i=1:nq
    set_minimal_coordinates!(mech,joints[i],[qpos[i]])
    set_minimal_velocities!(mech,joints[i],[0])
end


u = zeros(nq)
x = get_minimal_state(mech)
xT = [0.009554512308719641;
0;
-0.38691447420381925;
0;
0.11517211055980892;
0;
-1.9922714117129157;
0;
0.01194013722382598;
0;
1.5828848796160164;
0;
-0.681276499563443;
0;
0.00018662022830896464;
0;
0.0006071448903509512;
0]
z = get_maximal_state(mech)

A, B = get_minimal_gradients!(mech, z, u)

Q = diagm([10000; 100; 10000; 100; 10000; 100; 10000; 100; 10000; 100; 10000; 100; 10000; 100; 10000; 100; 10000; 100])
# Q = diagm([10000; 100; 10000; 100; 10000; 100; 10000; 100; 10000; 100; 10000; 100])

R = diagm([0.001; 0.001; 0.001; 0.01; 0.01; 0.01; 0.01; 1; 1])
# R = diagm([0.001; 0.001; 0.001; 0.01; 0.01; 0.01])

function rec(A,B,Q,R,N)
    P = Q
    K = [zeros(nq,nq*2) for i=1:N]

    for i=N:-1:1
        display(i)
        K[i] = -(R+B'*P*B)\B'*P*A
        Abar = A-B*K[i]
        P = Q + K[i]'*R*K[i]+Abar'*P*Abar
    end

    return K
end

# myK = rec(A,B,Q,R,250)[1]

myK = lqr(Discrete,A,B,Q,R)

function controller!(mechanism, k)
    global cost
    u = myK*(x-get_minimal_state(mech))
    for j in u
        write(f, string(j))
        write(f, '\n')
    end
    for i=1:nq-2
        # cost += dot(u, u)
        if k == 10
            cost += 1*((minimal_coordinates(mech, joints[i])[1] - x[i])^2)
        else
            cost += 1*((minimal_coordinates(mech, joints[i])[1] - x[i])^2)
        end
        set_input!(joints[i],[u[i]])
    end
end

mech.gravity = [0;0;0]
f = open("ctrl.txt", "w")
cost = 0
storage = simulate!(mech, 10*dt, controller!, record=true)
close(f)
print("Cost", cost)
visualize(mech, storage)[1]