import numpy as np
import torch

from util.mesh import Mesh, HalfedgeMesh

def main():
    mesh = Mesh("data/box.obj")
    he_mesh = HalfedgeMesh(mesh)
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()