class Vertex:
    def __init__(self, x=0, y=0, z=0, index=None, halfedge=None):
        self.x = x
        self.y = y
        self.z = z
        self.index = index
        self.halfedge = halfedge

class Face:
    def __init__(self, a=-1, b=-1, c=-1, index=None, halfedge=None):
        self.a = a
        self.b = b
        self.c = c
        self.index = index
        self.halfedge = halfedge

class Halfedge:
    def __init__(self, vertex=None, face=None, pair=None, next=None, prev=None, index=None):
        self.vertex = vertex
        self.face = face
        self.pair = pair
        self.next = next
        self.prev = prev
        self.index = index

        if vertex.halfedge == None:
            vertex.halfedge = self

class HalfedgeMesh:
    def __init__(self, mesh):
        self.faces = []
        self.vertices = []
        self.halfedge = []
        self.mesh = mesh
        self.mesh_to_halfedgeMesh(self.mesh.vs, self.mesh.faces)

    
    def set_halfedge_pair(self, he):
        for i in range(len(self.faces)):
            he_in_face = self.faces[i].halfedge
            while True:
                if he.vertex == he_in_face.next.vertex and he.next.vertex == he_in_face.vertex:
                    he.pair = he_in_face
                    he_in_face.pair = he
                    return
                he_in_face = he_in_face.next
                
                if he_in_face == self.faces[i].halfedge:
                    break
    
    def addface(self, v0, v1, v2):
        he0 = Halfedge(vertex=v0)
        he1 = Halfedge(vertex=v1)
        he2 = Halfedge(vertex=v2)

        he0.next, he0.prev = he1, he2
        he1.next, he1.prev = he2, he0
        he2.next, he2.prev = he0, he1

        face = Face(halfedge=he0)
        self.faces.append(face)
        
        he0.face = face
        he1.face = face
        he2.face = face
        self.set_halfedge_pair(he0)
        self.set_halfedge_pair(he1)
        self.set_halfedge_pair(he2)

    def mesh_to_halfedgeMesh(self, vs, faces):
        for v_i in vs:
            v = Vertex(x=v_i[0], y=v_i[1], z=v_i[2])
            self.vertices.append(v)
        
        for f_i in faces:
            self.addface(self.vertices[f_i[0]], self.vertices[f_i[1]], self.vertices[f_i[2]])