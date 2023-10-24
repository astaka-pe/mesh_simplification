import argparse
import os

from util.mesh import Mesh

def get_parser():
    parser = argparse.ArgumentParser(description="Mesh Simplification")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file name")
    parser.add_argument("-v", type=int, help="Target vertex number")
    parser.add_argument("-p", type=float, default=0.5, help="Rate of simplification (Ignored by -v)")
    parser.add_argument("-optim", action="store_true", help="Specify for valence aware simplification")
    parser.add_argument("-isotropic", action="store_true", help="Specify for Isotropic simplification")
    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    mesh = Mesh(args.input)
    mesh_name = os.path.basename(args.input).split(".")[-2]
    if args.v:
        target_v = args.v
    else:
        target_v = int(len(mesh.vs) * args.p)
    if target_v >= mesh.vs.shape[0]:
        print("[ERROR]: Target vertex number should be smaller than {}!".format(mesh.vs.shape[0]))
        exit()
    if args.isotropic:
        simp_mesh = mesh.edge_based_simplification(target_v=target_v, valence_aware=args.optim)
    else:
        simp_mesh = mesh.simplification(target_v=target_v, valence_aware=args.optim)
    os.makedirs("data/output/", exist_ok=True)
    simp_mesh.save("data/output/{}_{}.obj".format(mesh_name, simp_mesh.vs.shape[0]))
    print("[FIN] Simplification Completed!")

if __name__ == "__main__":
    main()