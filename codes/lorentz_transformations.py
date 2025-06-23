import numpy as np
from scipy.linalg import expm

Jz = np.array([[0, 0, 0, 0],
               [0, 0, -1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0]])
Jy = np.array([[0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 0],
               [0, -1, 0, 0]])

def lorentz_transform(P: np.ndarray) -> np.ndarray:
    """
        Construct a lorentz transformation that maps into a frame in which a 
        time-like future-directed four-vector P is at rest. This is done by 
        performing a boost in the direction of the spatial component of P.
        This is explained in Scheck's mechanics textbook.

        Args:
            P (np.ndarray): A 4-vector representing the four-momentum of a 
            particle.

        Returns:
            (np.ndarray): A 4x4 Lorentz transformation matrix
    """

    D = len(P)
    Pvec = P[1:]
    Ecm = np.sqrt(P[0]**2 - Pvec @ Pvec)
    beta = Pvec / P[0]
    gamma = P[0] / Ecm

    Lambda = np.zeros((4, 4))
    Lambda[0, 0] = gamma
    Lambda[0, 1:] = -gamma * beta
    Lambda[1:, 0] = -gamma * beta
    Lambda[1:, 1:] = np.eye(D - 1) + gamma**2 * np.outer(beta, beta) / (1 + gamma)

    return Lambda

def rotation(P: np.ndarray) -> np.ndarray:
    """
        Construct a rotation that maps into a frame in which the four-vector P
        has a spatial component along the z direction. This rotation is 
        obtained by first rotating along the z axis so that the spatial 
        component is in the xy plane and then rotating along the y axis.

        Args:
            P (np.ndarray): A 4-vector representing the four-momentum of a 
            particle.

        Returns:
            (np.ndarray): A 4x4 Lorentz transformation matrix
    """

    Pvec = P[1:]

    theta = np.arctan2(np.sqrt(Pvec[0]**2 + Pvec[1]**2), Pvec[2])
    phi = np.arctan2(Pvec[1], Pvec[0])

    Rz = np.array([[1, 0, 0, 0],
                   [0, np.cos(phi), np.sin(phi), 0],
                   [0, -np.sin(phi), np.cos(phi), 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), 0, -np.sin(theta)],
                   [0, 0, 0, 0],
                   [0, np.sin(theta), 0, np.cos(theta)]])

    return Ry @ Rz

def lorentz_transformation_2(Pi: np.ndarray, Pf: np.ndarray) -> np.ndarray:
    """
        Constructs a Lorentz transformation that maps into a reference frame
        in which the four-vector Pi is at rest and the four-vector Pf has a 
        spatial component along the z-axis. This is not unique but is obtained
        through the composition of the two functions above.

        Args:
            Pi (np.ndarray): A 4-vector representing the initial four-momentum
            of a particle.
            Pf (np.ndarray): A 4-vector representing the final four-momentum
            of a particle.

        Returns:
            (np.ndarray): A 4x4 Lorentz transformation matrix
    """
    L = lorentz_transform(Pi)
    return rotation(L @ Pf) @ L

# Pi = np.array([3, 0, 0, 1])
# Pf = np.array([5, 1, 1, 1])
# L = lorentz_transformation_2(Pi, Pf)
# print(L @ Pf)
# print(L @ Pi, np.sqrt(Pi[0]**2 - np.linalg.norm(Pi[1:])**2))
