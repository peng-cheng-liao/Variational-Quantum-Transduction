

from main import *

statetype = "_coherent_"
Nt = 50

state_signals = [coherent_state(1, Nt)]
probabilities = [1]
theta = np.pi / 10

e = 1.0
r = np.arcsinh(np.sqrt(e))
G = np.cosh(r) ** 2
eta = np.sin(theta) ** 2
Gprime = G / (G * eta + 1 - eta)
rprime = torch.abs(torch.arccosh(torch.sqrt(torch.tensor(Gprime))))
rho0, n1 = transduction_protocol_TMS(state_signals[0], theta, r, -rprime, Nt)
f0 = state_fidelity(rho0, state_signals[0])
d0 = trace_distance(rho0, state_signals[0])
data = [r, G, rprime.detach().item(), n1.detach().item(), eta, f0.detach().item(), d0.detach().item(),]
np.save("Data/Transduction_TMS" + statetype + f"Nt={Nt}_r={np.around(r,2)}", data)
print(data)




