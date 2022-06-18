from util.mesh import Mesh

def main():
    mesh = Mesh("data/ankylosaurus.obj")
    v1 = int(len(mesh.vs) * 0.2)
    simp_mesh = mesh.simplification(target_v=v1)
    simp_mesh.save("data/ankylosaurus_{}.obj".format(len(simp_mesh.vs)))

if __name__ == "__main__":
    main()