from util.mesh import Mesh

def main():
    mesh = Mesh("data/ankylosaurus.obj")
    v1 = int(len(mesh.vs) * 0.8)
    v2 = int(len(mesh.vs) * 0.6)
    simp_mesh = mesh.simplification(target_v=v1)
    simp_mesh.save("data/ankylosaurus_{}.obj".format(v1))
    simp_mesh = mesh.simplification(target_v=v2)
    simp_mesh.save("data/ankylosaurus_{}.obj".format(v2))

if __name__ == "__main__":
    main()