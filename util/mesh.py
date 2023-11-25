import numpy as np
import scipy as sp
import heapq
import copy
from tqdm import tqdm
from sklearn.preprocessing import normalize

OPTIM_VALENCE = 6
VALENCE_WEIGHT = 1

class Mesh:
    def __init__(self, path, build_code=False, build_mat=False, manifold=True):
        self.path = path
        self.vs, self.faces = self.fill_from_file(path)
        self.compute_face_normals()
        self.compute_face_center()
        self.device = 'cpu'
        self.simp = False
        
        if manifold:
            self.build_gemm() #self.edges, self.ve
            self.compute_vert_normals()
            self.build_v2v()
            self.build_vf()
            self.build_uni_lap()

    def fill_from_file(self, path):
        vs, faces = [], []
        f = open(path)
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind) for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
        f.close()
        vs = np.asarray(vs)
        faces = np.asarray(faces, dtype=int)

        assert np.logical_and(faces >= 0, faces < len(vs)).all()
        return vs, faces

    def build_gemm(self):
        self.ve = [[] for _ in self.vs]
        self.vei = [[] for _ in self.vs]
        edge_nb = []
        sides = []
        edge2key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        for face_id, face in enumerate(self.faces):
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    self.ve[edge[0]].append(edges_count)
                    self.ve[edge[1]].append(edges_count)
                    self.vei[edge[0]].append(0)
                    self.vei[edge[1]].append(1)
                    nb_count.append(0)
                    edges_count += 1
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
        self.edges = np.array(edges, dtype=np.int32)
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count

    def compute_face_normals(self):
        face_normals = np.cross(self.vs[self.faces[:, 1]] - self.vs[self.faces[:, 0]], self.vs[self.faces[:, 2]] - self.vs[self.faces[:, 0]])
        norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24
        face_areas = 0.5 * np.sqrt((face_normals**2).sum(axis=1))
        face_normals /= norm
        self.fn, self.fa = face_normals, face_areas

    def compute_vert_normals(self):
        vert_normals = np.zeros((3, len(self.vs)))
        face_normals = self.fn
        faces = self.faces

        nv = len(self.vs)
        nf = len(faces)
        mat_rows = faces.reshape(-1)
        mat_cols = np.array([[i] * 3 for i in range(nf)]).reshape(-1)
        mat_vals = np.ones(len(mat_rows))
        f2v_mat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(nv, nf))
        vert_normals = sp.sparse.csr_matrix.dot(f2v_mat, face_normals)
        vert_normals = normalize(vert_normals, norm='l2', axis=1)
        self.vn = vert_normals
    
    def compute_face_center(self):
        faces = self.faces
        vs = self.vs
        self.fc = np.sum(vs[faces], 1) / 3.0
    
    def build_uni_lap(self):
        """compute uniform laplacian matrix"""
        edges = self.edges
        ve = self.ve

        sub_mesh_vv = [edges[v_e, :].reshape(-1) for v_e in ve]
        sub_mesh_vv = [set(vv.tolist()).difference(set([i])) for i, vv in enumerate(sub_mesh_vv)]

        num_verts = self.vs.shape[0]
        mat_rows = [np.array([i] * len(vv), dtype=np.int64) for i, vv in enumerate(sub_mesh_vv)]
        mat_rows = np.concatenate(mat_rows)
        mat_cols = [np.array(list(vv), dtype=np.int64) for vv in sub_mesh_vv]
        mat_cols = np.concatenate(mat_cols)
        mat_vals = np.ones_like(mat_rows, dtype=np.float32) * -1.0
        neig_mat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(num_verts, num_verts))
        sum_count = sp.sparse.csr_matrix.dot(neig_mat, np.ones((num_verts, 1), dtype=np.float32))

        mat_rows_ident = np.array([i for i in range(num_verts)])
        mat_cols_ident = np.array([i for i in range(num_verts)])
        mat_ident = np.array([-s for s in sum_count[:, 0]])


        mat_rows = np.concatenate([mat_rows, mat_rows_ident], axis=0)
        mat_cols = np.concatenate([mat_cols, mat_cols_ident], axis=0)
        mat_vals = np.concatenate([mat_vals, mat_ident], axis=0)

        self.lapmat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(num_verts, num_verts))

    def build_vf(self):
        vf = [set() for _ in range(len(self.vs))]
        for i, f in enumerate(self.faces):
            vf[f[0]].add(i)
            vf[f[1]].add(i)
            vf[f[2]].add(i)
        self.vf = vf

    def build_v2v(self):
        v2v = [[] for _ in range(len(self.vs))]
        for i, e in enumerate(self.edges):
            v2v[e[0]].append(e[1])
            v2v[e[1]].append(e[0])
        self.v2v = v2v

        """ compute adjacent matrix """
        edges = self.edges
        v2v_inds = edges.T
        v2v_inds = np.concatenate([v2v_inds, v2v_inds[[1, 0]]], axis=1).astype(np.int64)
        v2v_vals = np.ones(v2v_inds.shape[1], dtype=np.float32)
        self.v2v_mat = sp.sparse.csr_matrix((v2v_vals, v2v_inds), shape=(len(self.vs), len(self.vs)))
        self.v_dims = np.sum(self.v2v_mat.toarray(), axis=1)

    def simplification(self, target_v, valence_aware=True, midpoint=False):
        vs, vf, fn, fc, edges = self.vs, self.vf, self.fn, self.fc, self.edges

        """ 1. compute Q for each vertex """
        Q_s = [[] for _ in range(len(vs))]
        E_s = [[] for _ in range(len(vs))]
        for i, v in enumerate(vs):
            f_s = np.array(list(vf[i]))
            fc_s = fc[f_s]
            fn_s = fn[f_s]
            d_s = - 1.0 * np.sum(fn_s * fc_s, axis=1, keepdims=True)
            abcd_s = np.concatenate([fn_s, d_s], axis=1)
            Q_s[i] = np.matmul(abcd_s.T, abcd_s)
            v4 = np.concatenate([v, np.array([1])])
            E_s[i] = np.matmul(v4, np.matmul(Q_s[i], v4.T))

        """ 2. compute E for every possible pairs and create heapq """
        E_heap = []
        for i, e in enumerate(edges):
            v_0, v_1 = vs[e[0]], vs[e[1]]
            Q_0, Q_1 = Q_s[e[0]], Q_s[e[1]]
            Q_new = Q_0 + Q_1

            if midpoint:
                v_new = 0.5 * (v_0 + v_1)
                v4_new = np.concatenate([v_new, np.array([1])])
            else:
                Q_lp = np.eye(4)
                Q_lp[:3] = Q_new[:3]
                try:
                    Q_lp_inv = np.linalg.inv(Q_lp)
                    v4_new = np.matmul(Q_lp_inv, np.array([[0,0,0,1]]).reshape(-1,1)).reshape(-1)
                except:
                    v_new = 0.5 * (v_0 + v_1)
                    v4_new = np.concatenate([v_new, np.array([1])])

            valence_penalty = 1
            if valence_aware:
                merged_faces = vf[e[0]].intersection(vf[e[1]])
                valence_new = len(vf[e[0]].union(vf[e[1]]).difference(merged_faces))
                valence_penalty = self.valence_weight(valence_new)
            
            
            E_new = np.matmul(v4_new, np.matmul(Q_new, v4_new.T)) * valence_penalty
            heapq.heappush(E_heap, (E_new, (e[0], e[1])))
        
        """ 3. collapse minimum-error vertex """
        simp_mesh = copy.deepcopy(self)

        vi_mask = np.ones([len(simp_mesh.vs)]).astype(np.bool_)
        fi_mask = np.ones([len(simp_mesh.faces)]).astype(np.bool_)

        vert_map = [{i} for i in range(len(simp_mesh.vs))]
        pbar = tqdm(total=np.sum(vi_mask)-target_v, desc="Processing")
        while np.sum(vi_mask) > target_v:
            if len(E_heap) == 0:
                print("[Warning]: edge cannot be collapsed anymore!")
                break

            E_0, (vi_0, vi_1) = heapq.heappop(E_heap)

            if (vi_mask[vi_0] == False) or (vi_mask[vi_1] == False):
                continue

            """ edge collapse """
            shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])

            if len(shared_vv) != 2:
                """ non-manifold! """
                # print("non-manifold can be occured!!" , len(shared_vv))
                self.remove_tri_valance(simp_mesh, vi_0, vi_1, shared_vv, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap)
                continue

            elif len(merged_faces) != 2:
                """ boundary """
                # print("boundary edge cannot be collapsed!")
                continue

            else:
                self.edge_collapse(simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap, valence_aware=valence_aware)
                pbar.update(1)
                # print(np.sum(vi_mask), np.sum(fi_mask))
        
        self.rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map)
        simp_mesh.simp = True
        self.build_hash(simp_mesh, vi_mask, vert_map)
        
        return simp_mesh
    
    def edge_based_simplification(self, target_v, valence_aware=True):
        vs, vf, fn, fc, edges = self.vs, self.vf, self.fn, self.fc, self.edges
        edge_len = vs[edges][:,0,:] - vs[edges][:,1,:]
        edge_len = np.linalg.norm(edge_len, axis=1)
        edge_len_heap = np.stack([edge_len, np.arange(len(edge_len))], axis=1).tolist()
        heapq.heapify(edge_len_heap)

        """ 2. compute E for every possible pairs and create heapq """
        E_heap = []
        for i, e in enumerate(edges):
            v_0, v_1 = vs[e[0]], vs[e[1]]
            heapq.heappush(E_heap, (edge_len[i], (e[0], e[1])))
        
        """ 3. collapse minimum-error vertex """
        simp_mesh = copy.deepcopy(self)

        vi_mask = np.ones([len(simp_mesh.vs)]).astype(np.bool_)
        fi_mask = np.ones([len(simp_mesh.faces)]).astype(np.bool_)

        vert_map = [{i} for i in range(len(simp_mesh.vs))]
        pbar = tqdm(total=np.sum(vi_mask)-target_v, desc="Processing")
        while np.sum(vi_mask) > target_v:
            if len(E_heap) == 0:
                print("[Warning]: edge cannot be collapsed anymore!")
                break

            E_0, (vi_0, vi_1) = heapq.heappop(E_heap)

            if (vi_mask[vi_0] == False) or (vi_mask[vi_1] == False):
                continue

            """ edge collapse """
            shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])

            if len(shared_vv) != 2:
                """ non-manifold! """
                # print("non-manifold can be occured!!" , len(shared_vv))
                continue

            elif len(merged_faces) != 2:
                """ boundary """
                # print("boundary edge cannot be collapsed!")
                continue

            else:
                self.edge_based_collapse(simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, E_heap, valence_aware=valence_aware)
                pbar.update(1)
                # print(np.sum(vi_mask), np.sum(fi_mask))
        
        self.rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map)
        simp_mesh.simp = True
        self.build_hash(simp_mesh, vi_mask, vert_map)
        
        return simp_mesh
    
    @staticmethod
    def remove_tri_valance(simp_mesh, vi_0, vi_1, shared_vv, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap):
        #import pdb;pdb.set_trace()
        pass  
    
    def edge_collapse(self, simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap, valence_aware):
        shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
        new_vi_0 = set(simp_mesh.v2v[vi_0]).union(set(simp_mesh.v2v[vi_1])).difference({vi_0, vi_1})
        simp_mesh.vf[vi_0] = simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        simp_mesh.vf[vi_1] = set()
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0}))
        simp_mesh.v2v[vi_1] = []
        vi_mask[vi_1] = False

        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()
        
        fi_mask[np.array(list(merged_faces)).astype(np.int32)] = False

        simp_mesh.vs[vi_0] = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vi_1])

        """ recompute E """
        Q_0 = Q_s[vi_0]
        for vv_i in simp_mesh.v2v[vi_0]:
            v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])
            Q_1 = Q_s[vv_i]
            Q_new = Q_0 + Q_1
            v4_mid = np.concatenate([v_mid, np.array([1])])

            valence_penalty = 1
            if valence_aware:
                merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vv_i])
                valence_new = len(simp_mesh.vf[vi_0].union(simp_mesh.vf[vv_i]).difference(merged_faces))
                valence_penalty = self.valence_weight(valence_new)

            E_new = np.matmul(v4_mid, np.matmul(Q_new, v4_mid.T)) * valence_penalty
            heapq.heappush(E_heap, (E_new, (vi_0, vv_i)))

    def edge_based_collapse(self, simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, E_heap, valence_aware):
        shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
        new_vi_0 = set(simp_mesh.v2v[vi_0]).union(set(simp_mesh.v2v[vi_1])).difference({vi_0, vi_1})
        simp_mesh.vf[vi_0] = simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        simp_mesh.vf[vi_1] = set()
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0}))
        simp_mesh.v2v[vi_1] = []
        vi_mask[vi_1] = False

        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()
        
        fi_mask[np.array(list(merged_faces)).astype(np.int32)] = False

        simp_mesh.vs[vi_0] = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vi_1])

        """ recompute E """
        for vv_i in simp_mesh.v2v[vi_0]:
            v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])
            edge_len = np.linalg.norm(simp_mesh.vs[vi_0] - simp_mesh.vs[vv_i])
            valence_penalty = 1
            if valence_aware:
                merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vv_i])
                valence_new = len(simp_mesh.vf[vi_0].union(simp_mesh.vf[vv_i]).difference(merged_faces))
                valence_penalty = self.valence_weight(valence_new)
                edge_len *= valence_penalty

            heapq.heappush(E_heap, (edge_len, (vi_0, vv_i)))

    @staticmethod
    def valence_weight(valence_new):
        valence_penalty = abs(valence_new - OPTIM_VALENCE) * VALENCE_WEIGHT + 1
        if valence_new == 3:
            valence_penalty *= 100000
        return valence_penalty      
    
    @staticmethod
    def rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map):
        face_map = dict(zip(np.arange(len(vi_mask)), np.cumsum(vi_mask)-1))
        simp_mesh.vs = simp_mesh.vs[vi_mask]
        
        vert_dict = {}
        for i, vm in enumerate(vert_map):
            for j in vm:
                vert_dict[j] = i

        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                if f[j] in vert_dict:
                    simp_mesh.faces[i][j] = vert_dict[f[j]]

        simp_mesh.faces = simp_mesh.faces[fi_mask]
        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                simp_mesh.faces[i][j] = face_map[f[j]]
        
        simp_mesh.compute_face_normals()
        simp_mesh.compute_face_center()
        simp_mesh.build_gemm()
        simp_mesh.compute_vert_normals()
        simp_mesh.build_v2v()
        simp_mesh.build_vf()

    @staticmethod
    def build_hash(simp_mesh, vi_mask, vert_map):
        pool_hash = {}
        unpool_hash = {}
        for simp_i, idx in enumerate(np.where(vi_mask)[0]):
            if len(vert_map[idx]) == 0:
                print("[ERROR] parent node cannot be found!")
                return
            for org_i in vert_map[idx]:
                pool_hash[org_i] = simp_i
            unpool_hash[simp_i] = list(vert_map[idx])
        
        """ check """
        vl_sum = 0
        for vl in unpool_hash.values():
            vl_sum += len(vl)

        if (len(set(pool_hash.keys())) != len(vi_mask)) or (vl_sum != len(vi_mask)):
            print("[ERROR] Original vetices cannot be covered!")
            return
        
        pool_hash = sorted(pool_hash.items(), key=lambda x:x[0])
        simp_mesh.pool_hash = pool_hash
        simp_mesh.unpool_hash = unpool_hash
            
    def save(self, filename):
        assert len(self.vs) > 0
        vertices = np.array(self.vs, dtype=np.float32).flatten()
        indices = np.array(self.faces, dtype=np.uint32).flatten()

        with open(filename, 'w') as fp:
            # Write positions
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write('v {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # Write indices
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0] + 1
                i1 = indices[i + 1] + 1
                i2 = indices[i + 2] + 1
                fp.write('f {0} {1} {2}\n'.format(i0, i1, i2))
    
    def save_as_ply(self, filename, fn):
        assert len(self.vs) > 0
        vertices = np.array(self.vs, dtype=np.float32).flatten()
        indices = np.array(self.faces, dtype=np.uint32).flatten()
        fnormals = np.array(fn, dtype=np.float32).flatten()

        with open(filename, 'w') as fp:
            # Write Header
            fp.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(self.vs)))
            fp.write("property float x\nproperty float y\nproperty float z\n")
            fp.write("element face {}\n".format(len(self.faces)))
            fp.write("property list uchar int vertex_indices\n")
            fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n")
            fp.write("end_header\n")
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write("{0:.6f} {1:.6f} {2:.6f}\n".format(x, y, z))
            
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0]
                i1 = indices[i + 1]
                i2 = indices[i + 2]
                c0 = fnormals[i + 0]
                c1 = fnormals[i + 1]
                c2 = fnormals[i + 2]
                c0 = np.clip(int(255 * c0), 0, 255)
                c1 = np.clip(int(255 * c1), 0, 255)
                c2 = np.clip(int(255 * c2), 0, 255)
                c3 = 255
                fp.write("3 {0} {1} {2} {3} {4} {5} {6}\n".format(i0, i1, i2, c0, c1, c2, c3))

    """ ---------- we don't use functions below ---------- """
    
    def pool_main(self, target_v):
        vs, vf, fn, fc, edges = self.vs, self.vf, self.fn, self.fc, self.edges

        """ 1. compute Q for each vertex """
        Q_s = [[] for _ in range(len(vs))]
        E_s = [[] for _ in range(len(vs))]
        for i, v in enumerate(vs):
            f_s = np.array(list(vf[i]))
            fc_s = fc[f_s]
            fn_s = fn[f_s]
            d_s = - 1.0 * np.sum(fn_s * fc_s, axis=1, keepdims=True)
            abcd_s = np.concatenate([fn_s, d_s], axis=1)
            Q_s[i] = np.matmul(abcd_s.T, abcd_s)

            v4 = np.concatenate([v, np.array([1])])
            E_s[i] = np.matmul(v4, np.matmul(Q_s[i], v4.T))

        """ 2. compute E for every possible pairs and create heapq """
        E_heap = []
        for i, e in enumerate(edges):
            v_0, v_1 = vs[e[0]], vs[e[1]]
            v_new = 0.5 * (v_0 + v_1)
            v4_new = np.concatenate([v_new, np.array([1])])
            
            Q_0, Q_1 = Q_s[e[0]], Q_s[e[1]]
            Q_new = Q_0 + Q_1
            E_new = np.matmul(v4_new, np.matmul(Q_new, v4_new.T))
            heapq.heappush(E_heap, (E_new, i))
        
        """ 3. collapse minimum-error vertex """

        mask = np.ones(self.edges_count, dtype=np.bool)
        self.v_mask = np.ones(len(self.vs), dtype=np.bool)
        while np.sum(self.v_mask) > target_v:
            E_0, edge_id = heapq.heappop(E_heap)
            edge = self.edges[edge_id]
            v_a = self.vs[edge[0]]
            v_b = self.vs[edge[1]]

            if mask[edge_id]:
                pool = self.pool_edge(edge_id, mask)
                if pool:
                    Q_0 = Q_s[edge[0]]
                    print(np.sum(self.v_mask), np.sum(mask))

                    """ recompute E """
                    for vv_i in self.v2v[edge[0]]:
                        Q_1 = Q_s[vv_i]
                        Q_new = Q_0 + Q_1
                        v4_mid = np.concatenate([self.vs[edge[0]], np.array([1])])
                        E_new = np.matmul(v4_mid, np.matmul(Q_new, v4_mid.T))
                        heapq.heappush(E_heap, (E_new, edge_id))
        self.clean(mask)

    def pool_edge(self, edge_id, mask):
        if self.has_boundaries(edge_id):
            return False
        elif self.is_one_ring_valid(edge_id):
            self.merge_vertices(edge_id)
            mask[edge_id] = False
            self.edges_count -= 1
            return True
        else:
            return False

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        self.vs[edge[0]] = 0.5 * (v_a + v_b)
        self.v_mask[edge[1]] = False
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]

    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]
        for v in vs:
            self.ve[v].remove(edge_id)
    
    def clean(self, edges_mask):
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            # if self.v_mask[v_index]:
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve

    def has_boundaries(self, edge_id):
        for edge in self.gemm_edges[edge_id]:
            if edge == -1 or -1 in self.gemm_edges[edge]:
                print(edge_id, "is boundary")
                return True
        return False

    def is_one_ring_valid(self, edge_id):
        v_a = set(self.edges[self.ve[self.edges[edge_id, 0]]].reshape(-1))
        v_b = set(self.edges[self.ve[self.edges[edge_id, 1]]].reshape(-1))
        shared = v_a.intersection(v_b).difference(set(self.edges[edge_id]))
        return len(shared) == 2

    def __get_cycle(self, gemm, edge_id):
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    def __cycle_to_face(self, cycle, v_indices):
        face = []
        for i in range(3):
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face