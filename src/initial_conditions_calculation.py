import numpy as np

class compute_initial_conditions():

    def __init__(self, Voltages, Hi, Rsi, Xdi, Xd_pi, Xqi, Xq_pi, Y_adm, simplification) -> None:

        self.V    = np.array(Voltages)
        self.H    = np.array(Hi)
        self.Rs   = np.array(Rsi)
        self.Xd   = np.array(Xdi)
        self.Xd_p = np.array(Xd_pi )
        self.Xq   = np.array(Xqi)
        self.Xq_p = np.array(Xq_pi)
        self.Yadmittance = Y_adm

        if simplification:
            self.Xq_p = self.Xd_p
            self.Xq   = self.Xd_p

    def compute_power_flow(self) -> tuple:
        currents = self.Yadmittance @ self.V

        powers = self.V * np.conjugate(currents)

        return np.real(powers), np.imag(powers)

    def current_network(self, P, Q) -> tuple:
        current_gen = np.conjugate((P+1j*Q)/self.V)
        return np.real(current_gen), np.imag(current_gen)

    def compute_delta(self, ID, IQ) -> tuple:
        phasor = self.V + (self.Rs+1j*self.Xq)*(ID+1j*IQ)
        omega = [0.0, 0.0, 0.0]
        return np.angle(phasor, deg=False), omega

    def compute_local_magns(self, ID, IQ, delta) -> tuple:
        currents_gen = (ID+1j*IQ)*np.exp(-1j*(delta-np.pi/2))
        voltages_gen = self.V*np.exp(-1j*(delta-np.pi/2))
        return np.real(currents_gen), np.imag(currents_gen), np.real(voltages_gen), np.imag(voltages_gen)

    def compute_Ed_prime(self, Id, Iq, Vd) -> float:
        Ed_p = (self.Xq - self.Xq_p)*Iq
        Ed_p_ver = Vd + self.Rs*Id - self.Xq_p*Iq
        if np.mean(Ed_p-Ed_p_ver) < 0.01:
            return Ed_p
        else:
            print(Ed_p, Ed_p_ver)
            raise("Error computing Ed_prime. Verification not passed")

    def compute_Eq_prime(self, Id, Iq, Vq) -> float:
        return Vq+self.Rs*Iq+self.Xd_p*Id

    def compute_Efd(self, Id, Eq_p) -> float:
        return Eq_p + (self.Xd-self.Xd_p)*Id
    
    def compute_initial_conditions(self, volt) -> list:
        pgi, qgi = self.compute_power_flow()
        IDgen, IQgen = self.current_network(pgi, qgi)
        deltas, omegas = self.compute_delta(IDgen, IQgen)
        Id_gen, Iq_gen, Vd_gen, Vq_gen = self.compute_local_magns(IDgen, IQgen, deltas)
        Ed_p = self.compute_Ed_prime(Id_gen, Iq_gen, Vd_gen)
        Eq_p = self.compute_Eq_prime(Id_gen, Iq_gen, Vq_gen)
        Efd = self.compute_Efd(Id_gen, Eq_p)

        array1 = [Eq_p[0], Ed_p[0], deltas[0], omegas[0], Id_gen[0], Iq_gen[0], IDgen[0], IQgen[0], np.abs(volt[0]), np.angle(volt[0])]
        array2 = [Eq_p[1], Ed_p[1], deltas[1], omegas[1], Id_gen[1], Iq_gen[1], IDgen[1], IQgen[1], np.abs(volt[1]), np.angle(volt[1])]
        array3 = [Eq_p[2], Ed_p[2], deltas[2], omegas[2], Id_gen[2], Iq_gen[2], IDgen[2], IQgen[2], np.abs(volt[2]), np.angle(volt[2])]

        return array1 + array2 + array3
