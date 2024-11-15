# ---------------------------------------------------------------------------
# Title: Func_pyrough
# Authors: Jonathan Amodeo, Javier Gonzalez, Jennifer Izaguirre, Christophe Le Bourlot, Hugo Iteney
# Date: June 01, 2022
#
# Func_pyrough are all the functions used in pyrough. These are required to execute the Pyrough
# main code.
# ---------------------------------------------------------------------------
import math

import gmsh
import meshio
import numpy as np
import numpy.typing as npt
import scipy.special as sp
from ase.build import bulk
from wulffpack import SingleCrystal
from scipy.spatial.transform import Rotation as R


NDArrayDouble = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int_]


# TODO make a Shape class for mesh generation ?
def initialize_gmsh_model(name: str) -> None:
    """
    Initializes a gmsh model and sets the terminal verbosity to 0.

    :param model_type: Name of the model
    :type model_type: str
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)


def generate_geo_mesh(dim: int = 2) -> tuple[NDArrayDouble, NDArrayInt]:
    """Generates the mesh current gmsh model and returns the vertices and faces.

    :return: vertices and faces of the mesh
    """
    gmsh.model.mesh.generate(dim)
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    element_types, _, element_nodes = gmsh.model.mesh.getElements(dim=2)

    triangle_idx = np.where(element_types == 2)[0][0]

    vertices = np.reshape(node_coords, (-1, 3))
    faces = np.reshape(element_nodes[triangle_idx], (-1, 3)) - 1

    return vertices, faces


def cylinder(length: float, r: float, ns: float) -> tuple[NDArrayDouble, NDArrayInt]:
    """
    :param length: Length of the cylinder
    :type length: float
    :param r: Radius of the cylinder
    :type r: float
    :param ns: Mesh size
    :type ns: float

    :return: vertices and faces of cylinder mesh
    """
    initialize_gmsh_model("Cylinder")

    # Add points to define the bottom circle
    center = gmsh.model.geo.addPoint(0, 0, 0)
    coords = [(r * np.cos(i * np.pi / 2), r * np.sin(i * np.pi / 2), 0.0) for i in range(4)]
    points = [gmsh.model.geo.addPoint(*coord) for coord in coords]

    # Create circle arcs
    arcs = [gmsh.model.geo.addCircleArc(points[i], center, points[(i + 1) % 4]) for i in range(4)]

    # Create bottom plane surface
    loop = gmsh.model.geo.addCurveLoop(arcs)
    disk = gmsh.model.geo.addPlaneSurface([loop])

    # Extrude the disk to get the cylinder
    gmsh.model.geo.extrude([(2, disk)], 0, 0, length, [length // ns])

    # Set a constant mesh size
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", str(ns))

    # Set the meshing field as the background field
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    gmsh.model.geo.synchronize()
    vertices, faces = generate_geo_mesh()
    gmsh.finalize()

    return vertices, faces


def box(width, length, height, ns):
    """
    :param width: Width of the box
    :type width: float
    :param length: Length of the box
    :type length: float
    :param height: Height of the box
    :type height: float
    :param ns: Mesh size
    :type ns: float

    :return: vertices and faces of box mesh
    """
    initialize_gmsh_model("Box")

    # Define the box's vertices
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(length, 0, 0)
    p3 = gmsh.model.geo.addPoint(length, width, 0)
    p4 = gmsh.model.geo.addPoint(0, width, 0)
    p5 = gmsh.model.geo.addPoint(0, 0, height)
    p6 = gmsh.model.geo.addPoint(length, 0, height)
    p7 = gmsh.model.geo.addPoint(length, width, height)
    p8 = gmsh.model.geo.addPoint(0, width, height)
    # Connect the points to form the edges
    lines = [
        gmsh.model.geo.addLine(p1, p2),
        gmsh.model.geo.addLine(p2, p3),
        gmsh.model.geo.addLine(p3, p4),
        gmsh.model.geo.addLine(p4, p1),
        gmsh.model.geo.addLine(p5, p6),
        gmsh.model.geo.addLine(p6, p7),
        gmsh.model.geo.addLine(p7, p8),
        gmsh.model.geo.addLine(p8, p5),
        gmsh.model.geo.addLine(p1, p5),
        gmsh.model.geo.addLine(p2, p6),
        gmsh.model.geo.addLine(p3, p7),
        gmsh.model.geo.addLine(p4, p8),
    ]
    # Connect the edges to form the faces
    faces = [
        gmsh.model.geo.addPlaneSurface(
            [gmsh.model.geo.addCurveLoop([lines[0], lines[1], lines[2], lines[3]])]
        ),
        gmsh.model.geo.addPlaneSurface(
            [gmsh.model.geo.addCurveLoop([lines[4], lines[5], lines[6], lines[7]])]
        ),
        gmsh.model.geo.addPlaneSurface(
            [gmsh.model.geo.addCurveLoop([lines[0], lines[9], -lines[4], -lines[8]])]
        ),
        gmsh.model.geo.addPlaneSurface(
            [gmsh.model.geo.addCurveLoop([-lines[2], lines[10], lines[6], -lines[11]])]
        ),
        gmsh.model.geo.addPlaneSurface(
            [gmsh.model.geo.addCurveLoop([lines[1], lines[10], -lines[5], -lines[9]])]
        ),
        gmsh.model.geo.addPlaneSurface(
            [gmsh.model.geo.addCurveLoop([-lines[3], lines[11], lines[7], -lines[8]])]
        ),
    ]
    # Define a uniform mesh size
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", str(ns))
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    gmsh.model.geo.synchronize()
    vertices, faces = generate_geo_mesh()
    gmsh.finalize()

    return vertices, faces


def sphere(r: float, ns: float) -> tuple[NDArrayDouble, NDArrayInt]:
    """
    :param r: Radius of the sphere
    :type r: float
    :param ns: Mesh size
    :type ns: float

    :return: vertices and faces of sphere mesh
    """
    initialize_gmsh_model("Sphere")

    gmsh.model.occ.addSphere(0, 0, 0, r)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", ns)
    gmsh.option.setNumber("Mesh.MeshSizeMax", ns)

    vertices, faces = generate_geo_mesh()
    gmsh.finalize()

    return vertices, faces


def poly(
    length: float, base_points: list[tuple[float, float]], ns: float
) -> tuple[NDArrayDouble, NDArrayInt]:
    """
    :param length: Length of the faceted wire
    :type length: float
    :param base_points: Shape of the base
    :type base_points: list
    :param ns: Mesh size
    :type ns: float

    :return: vertices and faces of faceted wire mesh
    """
    initialize_gmsh_model("Wire")

    points = [gmsh.model.geo.addPoint(p[0], p[1], 0, ns) for p in base_points]

    lines = [gmsh.model.geo.addLine(points[i], points[i + 1]) for i in range(len(points) - 1)]
    lines.append(gmsh.model.geo.addLine(points[-1], points[0]))

    curve_loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    gmsh.model.geo.extrude([(2, surface)], 0, 0, length, [length // ns])

    gmsh.model.geo.synchronize()
    vertices, faces = generate_geo_mesh()
    gmsh.finalize()

    return vertices, faces


def wulff(
    points: list[list[float]], facets: list[list[int]], ns: float
) -> tuple[NDArrayDouble, NDArrayInt]:
    """
    :param points: Vertices of Wulff-shape
    :type points: list
    :param faces: Facets of Wulff-shape
    :type faces: list
    :param ns: Mesh size
    :type ns: float

    :return: vertices and faces of Wulff mesh
    """
    initialize_gmsh_model("MeshFromPointsAndFaces")

    point_tags = [gmsh.model.geo.addPoint(p[0], p[1], p[2], ns) for p in points]

    for face in facets:
        line_loops = []
        for i in range(len(face)):
            start_point = point_tags[face[i] - 1]
            end_point = point_tags[face[(i + 1) % len(face)] - 1]

            line_loops.append(gmsh.model.geo.addLine(start_point, end_point))

        loop = gmsh.model.geo.addCurveLoop(line_loops)
        gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    vertices, faces = generate_geo_mesh()
    gmsh.finalize()

    return vertices, faces


def cube(length, ns):
    """
    :param length: Length of the cube
    :type length: float
    :param ns: Mesh size
    :type ns: float

    :return: vertices and faces of cube mesh
    """
    initialize_gmsh_model("Cube")

    # Add points for the base square
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(length, 0, 0)
    p3 = gmsh.model.geo.addPoint(length, length, 0)
    p4 = gmsh.model.geo.addPoint(0, length, 0)

    # Connect the points to form the base square
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Create a surface from the base square
    base_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    base_surface = gmsh.model.geo.addPlaneSurface([base_loop])

    # Extrude the base surface to get the cube
    gmsh.model.geo.extrude([(2, base_surface)], 0, 0, length)

    # Define a uniform mesh size
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", str(ns))
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    gmsh.model.geo.synchronize()
    vertices, faces = generate_geo_mesh()
    gmsh.finalize()

    return vertices, faces


def read_stl(sample_type, raw_stl, width, length, height, radius, ns, points):
    """
    Reads an input stl file or creates a new one if no input

    :param sample_type: Name of the sample
    :type sample_type: str
    :param raw_stl: Name of the input stl file
    :type raw_stl: str
    :param width: Width of the box
    :type width: float
    :param length: Length of the box/wire
    :type length: float
    :param height: Height of the box
    :type height: float
    :param radius: Radius of the wire/sphere
    :type radius: float
    :param ns: The number of segments desired
    :type ns: int
    :param points: List of points constituting the base (in case of faceted wire)
    :type points: list

    :return: List of points and faces
    """
    if raw_stl == "na":
        print("====== > Creating the Mesh")
        if sample_type == "box" or sample_type == "grain":
            vertices, faces = box(width, length, height, ns)
        elif sample_type == "wire":
            vertices, faces = cylinder(length, radius, ns)
        elif sample_type == "sphere":
            vertices, faces = sphere(radius, ns)
        elif sample_type == "poly":
            vertices, faces = poly(length, points, ns)
        elif sample_type == "cube":
            vertices, faces = cube(length, ns)
        else:
            raise ValueError("Invalid sample type")
        print("====== > Done creating the Mesh")
    else:
        mesh = meshio.read(raw_stl)
        vertices, faces = mesh.points, mesh.cells
        vertices = np.asarray(vertices)
        faces = np.asarray(faces[0][1])

    return vertices, faces


# TODO merge with read_stl
def read_stl_wulff(raw_stl, obj_points, obj_faces, ns):
    """
    Reads an input stl file or creates a new one if no input. Wulff case.

    :param raw_stl: Name of the input stl file
    :type raw_stl: str
    :param obj_points: List of points from OBJ file
    :type obj_points: list
    :param obj_faces: List of faces from OBJ file
    :type obj_faces: list
    :param ns: Mesh size
    :type ns: float

    :returns: List of points and faces
    """
    if raw_stl == "na":
        vertices, faces = wulff(obj_points, obj_faces, ns)
    else:
        mesh = meshio.read(raw_stl)
        vertices, faces = mesh.vertices, mesh.faces
    return vertices, faces


def make_obj(
    surfaces,
    energies,
    n_at,
    lattice_structure,
    lattice_parameter,
    material,
    orien_x,
    orien_z,
    out_pre,
):
    """
    Creates an OBJ file of a faceted NP. Stores the points and faces from this file.

    :param surfaces: List of surfaces for Wulff theory
    :type surfaces: list
    :param energies: List of energies associated to surfaces for Wulff theory
    :type energies: list
    :param n_at: Number of atoms
    :type n_at: int
    :param lattice_structure: Atomic structure of the sample
    :type lattice_structure: str
    :param lattice_parameter: Lattice parameter
    :type lattice_parameter: float
    :param material: Type of atom
    :type material: str
    :param orien_x: Orientation along x-axis
    :type orien_x: list
    :param orien_y: Orientation along y-axis
    :type orien_y: list
    :param orien_z: Orientation along z-axis
    :type orien_z: list
    :param out_pre: Prefix for output files
    :type out_pre: str

    :returns: List of points and faces of OBJ file
    """
    surface_energies = {tuple(surfaces[i]): float(energies[i]) for i in range(0, len(surfaces))}
    prim = bulk(material, crystalstructure=lattice_structure, a=lattice_parameter)
    particle = SingleCrystal(surface_energies, primitive_structure=prim, natoms=n_at)
    particle.write(out_pre + ".obj")

    with open(out_pre + ".obj") as f:
        lines = f.readlines()

    obj_points = []
    obj_faces = []
    for line in lines:
        parts = line.split()
        if "v" in line:
            obj_points.append([float(coord) for coord in parts[1:]])
        elif "f" in line:
            obj_faces.append([int(coord) for coord in parts[1:]])

    obj_points = np.asarray(obj_points)
    obj_points_f = rotate_obj_wulff(obj_points, orien_x, orien_z)

    return obj_points_f, obj_faces


def phi(t, z):
    """
    Calculates the arctan2 of an array filled with vector norms and an array filled with z
    coordinates which are the same size.

    :param t: Coordinates x y in a list
    :type t: array
    :param z: Z coordinates
    :type z: array

    :return: An array of angles in radians
    """
    return np.arctan2(np.linalg.norm(t, axis=1), z)


def vertex_tp(x, y, t, z):
    """
    Creates an array filled with two elements that are the angles corresponding to the position of
    the node on the sphere.

    :param x: X coordinates
    :type x: array
    :param y: Y coordinates
    :type y: array
    :param t: x & y coordinates stored in a list
    :type t: array
    :param z: Z coordinates
    :type z: array

    :return: An array with angles as elements
    """
    return np.array([np.arctan2(y, x), phi(t, z)]).T


def stat_analysis(z, N, M, C1, B, sample_type, out_pre):
    """
    Displays the statistical analysis of the surface roughness generator

    :param z: Roughness height matrix
    :type z: float
    :param N: Scaling cartesian position
    :type N: int
    :param M: Scaling cartesian position
    :type M: int
    :param C1: Roughness normalization factor
    :type C1: float
    :param B: The degree the roughness is dependent on
    :type B: float
    :param sample_type: The name of the sample
    :type sample_type: str
    :param out_pre: Prefix of output files
    :type out_pre: str

    """
    z_an = z.flatten()
    nu_points = len(z_an)
    mean = np.mean(z_an)
    stand = np.std(z_an)
    rms = np.sqrt(np.sum(np.square(z_an)) / len(z_an))
    skewness = np.sum(np.power((z_an - np.mean(z_an)), 3) / len(z_an)) / np.power(np.std(z_an), 3)
    kurtosis = np.sum(np.power((z_an - np.mean(z_an)), 4) / len(z_an)) / np.power(np.std(z_an), 4)

    stats = [
        N,
        M,
        C1,
        round(0.5 * B - 1, 2),
        nu_points,
        mean,
        stand,
        rms,
        skewness,
        kurtosis,
    ]
    stats = list(map(str, stats))
    stats = [
        sample_type,
        "N = " + stats[0],
        "M = " + stats[1],
        "C1 = " + stats[2],
        "eta = " + stats[3],
        "No. points = " + stats[4],
        "Mean_Value = " + stats[5],
        "Stand_dev = " + stats[6],
        "RMS = " + stats[7],
        "Skewness = " + stats[8],
        "Kurtosis = " + stats[9],
    ]

    np.savetxt(out_pre + "_stat.txt", stats, fmt="%s")
    print("")
    print("------------ Random Surface Parameters-----------------")
    print(f"         N = {N}  M = {M}  C1 = {C1}  eta = {round(0.5 * B - 1, 2)}")
    print(f"No. points = {nu_points}")
    print(f"Mean_Value = {mean}")
    print(f" Stand_dev = {stand}")
    print(f"       RMS = {rms}")
    print(f"  Skewness = {skewness}")
    print(f"  Kurtosis = {kurtosis}")
    print("--------------------------------------------------------")


def duplicate(side_length, orien, lattice_par):
    """
    Takes in a length and an crystal orientation to calculate the duplication factor for atomsk.

    :param side_length: Length of one of the sides of the object
    :type side_length: int
    :param orien: Crystal orientation
    :type orien: list
    :param lattice_par: Lattice parameter
    :type lattice_par: float

    :returns: Duplication factor and string of orientations
    """
    storage = [x**2 for x in orien]
    total = sum(storage)
    
    distance = lattice_par * math.sqrt(total)
    if total == 6 or total == 2:
        distance /= 2
    
    dup = math.ceil(side_length / distance)
        
    end_orien = "".join(map(str, orien))
    x = f"[{end_orien}]"
    
    return 0.5 * distance, str(dup), x


def random_numbers(sfrN, sfrM):
    """
    Generates the G and U matrices for the mathematical formulation of rough surfaces.

    :param sfrN: Vector for the N decomposition
    :type sfrN: array
    :param sfrM: Vector for the M decomposition
    :type sfrM: array

    :returns: G and U matrices
    """
    m = 0 + 1 * np.random.randn(len(sfrM), len(sfrN))  # Gaussian distributed
    n = -np.pi / 2 + np.pi * np.random.rand(len(sfrM), len(sfrN))  # Uniform distributed
    return m, n


def node_indexing(vertices):
    """
    Creates a column that has an assigned index number for each row in vertices and also the
    nodenumbers which is an array file from 0 - length of the vertices

    :param vertices: Vector for the points
    :type vertices: array

    :returns: Numbered vertices and raw numbers column.
    """
    nodenumber = range(0, len(vertices))
    vertices = np.insert(vertices, 3, nodenumber, 1)
    return vertices, nodenumber


def node_surface(sample_type, vertices, nodenumber, points, faces):
    """
    Finds the nodes at the surface of the object. These nodes will have the surface roughness
    applied to it.

    :param sample_type: The name of the sample
    :type sample_type: str
    :param vertices: List of nodes
    :type vertices: array
    :param nodenumber: Number of the corresponding node
    :type nodenumber: array
    :param points : Polygon shape (Faceted wire case)
    :type points: array
    :param faces: Facets list (Wulff case)
    type faces: array

    :return: Surface nodes
    """
    stay = []
    if sample_type == "wire":
        max_height = max(vertices[:, 1])
        for x in range(0, len(vertices)):
            if np.hypot(vertices[x, 0], vertices[x, 1]) > (max_height - 0.1):
                stay.append(x)
        no_need = np.delete(nodenumber, stay)  # delete from nodenumbers the ones in the surface
        nodesurf = np.delete(vertices, no_need, 0)
        return nodesurf

    elif sample_type == "box" or sample_type == "grain":
        eps = 0.000001
        max_height = max(vertices[:, 2])
        for index in range(0, len(vertices)):
            if abs(vertices[index][2] - max_height) <= eps:
                stay.append(index)
        no_need = np.delete(nodenumber, stay)  # delete from nodenumbers the ones in the surface
        nodesurf = np.delete(vertices, no_need, 0)
        return nodesurf

    elif sample_type == "poly":
        eps = 0.0001
        face = []
        for i in range(len(points)):
            k = i + 1
            if k == len(points):
                k = 0
            p1 = points[i]
            p2 = points[k]
            if p2[0] - p1[0] >= -eps and p2[0] - p1[0] <= eps:
                K = p2[0]
                for verti in vertices:
                    if verti[0] >= K - eps and verti[0] <= K + eps:
                        face = np.append(face, verti, axis=0)
            else:
                A = (p2[1] - p1[1]) / (p2[0] - p1[0])
                B = p2[1] - A * p2[0]
                for verti in vertices:
                    if A * verti[0] + B >= verti[1] - eps and A * verti[0] + B <= verti[1] + eps:
                        face = np.append(face, verti, axis=0)
        n_faces = len(face)
        nodesurf = np.reshape(face, [int(n_faces / 4), 4])
        return np.array(remove_duplicates_2d_ordered(nodesurf))

    elif sample_type == "wulff" or sample_type == "cube":
        nodesurf = []
        for F in faces:
            L = []
            eps = 0.000001
            A = points[int(F[0] - 1)]
            B = points[int(F[1] - 1)]
            C = points[int(F[2] - 1)]
            AB = B - A
            AC = C - A
            for M in vertices:
                AM = M[:3] - A
                matrix = np.array(
                    [
                        [AM[0], AB[0], AC[0]],
                        [AM[1], AB[1], AC[1]],
                        [AM[2], AB[2], AC[2]],
                    ],
                    dtype=float,
                )
                det = np.linalg.det(matrix)
                if abs(det) <= eps:
                    L.append([M[0], M[1], M[2], M[3]])
            nodesurf.append(L)
        return nodesurf


def remove_duplicates_2d_ordered(data):
    """

    :param data: Initial list
    :type data: list

    :return: List with no duplicates
    """
    seen = set()
    result = []
    
    for item in data:
        t_item = tuple(item)
        
        if t_item not in seen:
            result.append(item)
            seen.add(t_item)
    
    return result


def coord_sphere(vertices):
    """
    Creates a matrix with the columns that correspond to the coordinates of either x, y, z and t
    which contains x y z coordinates in an array

    :param vertices: List of nodes
    :type vertices: array

    :return: Coordinates
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    t = vertices[:, 0:2]
    return (x, y, z, t)


def rough_matrix_sphere(nbPoint, B, thetaa, phii, vert_phi_theta, r):
    """
    Creates the displacement values of the nodes on the surface of the sphere

    :param nbPoint: Number of points on the sphere
    :type nbPoint: int
    :param B: The degree the roughness is dependent on
    :type B: float
    :param thetaa: Arctan2 of x and y coordinates
    :type thetaa: array
    :param phii: Arctan2 of vector norms and z coordinates
    :type phii: array
    :param vert_phi_theta: Array filled with two elements that are the angles corresponding to the
    position of the node on the sphere.
    :type vert_phi_theta: array
    :param r: Roughness height matrix
    :type r: int

    :return: Rough matrix
    """
    N_s = 9
    N_e = 17
    print("====== > Creating random rough surface")
    for degree in range(N_s, N_e + 1, 1):  # EQUATION
        if degree == N_s:
            print(f"====== > Harmonic degree : {degree}, ", end=" ", flush=True)
        elif degree == N_e:
            print(f"{degree}.")
        else:
            print(f"{degree}, ", end=" ", flush=True)
        _r_amplitude = 0 + 1 * np.random.randn(nbPoint)
        _r_phase = -np.pi / 2 + np.pi * np.random.rand(nbPoint)
        mod = degree ** (-B / 2)
        for i, [theta, phi] in enumerate(vert_phi_theta):
            _phase = sp.sph_harm(0, degree, thetaa - theta, phii - phi).real
            _phase = 2 * _phase / np.ptp(_phase)
            r += _r_amplitude[i] * mod * np.cos(_phase + _r_phase)
    return r


def coord_cart_sphere(C1, C2, r, vertices, t, z, y, x):
    """
    Creates a new matrix with x, y, z in cartesian coordinates

    :param C1: Roughness normalization factor
    :type C1: float
    :param C2: Roughness normalization factor constant for sphere
    :type C2: float
    :param r: Roughness height matrix
    :type r: int
    :param vertices: List of nodes
    :type vertices: array
    :param t: x y z coordinates
    :type t: array
    :param z: z coordinates
    :type z: array
    :param y: y coordinates
    :type y: array
    :param x: x coordinates
    :type x: array

    :return: Cartesian coordinates
    """
    radius = np.linalg.norm(vertices, axis=1)
    phis = phi(t, z)
    thetas = np.arctan2(x, y)

    x = (C1 * C2 * r + radius) * np.sin(phis) * np.cos(thetas)
    y = (C1 * C2 * r + radius) * np.sin(phis) * np.sin(thetas)
    z = (C1 * C2 * r + radius) * np.cos(phis)

    return np.column_stack((x, y, z))


def rebox(file_lmp):
    """
    Fits the box dimensions with the sample. Also centers the sample.

    :param file_lmp: .lmp file containing the atom positions
    :type file_lmp: str

    :return: Reboxed position file
    """
    fint = open(file_lmp)
    lines = fint.readlines()
    data = np.array([i.split() for i in lines[15 : len(lines) : 1]])
    listN = data[:, 0]
    listi = data[:, 1]
    listx = data[:, 2]
    listy = data[:, 3]
    listz = data[:, 4]
    listN_int = [int(i) for i in listN]
    listi_int = [int(i) for i in listi]
    listx_float = [float(i) for i in listx]
    listy_float = [float(i) for i in listy]
    listz_float = [float(i) for i in listz]
    mx = min(listx_float)
    my = min(listy_float)
    mz = min(listz_float)
    listx_n = [x - mx for x in listx_float]
    listy_n = [y - my for y in listy_float]
    listz_n = [z - mz for z in listz_float]
    compt = 0
    for i in range(0, len(lines), 1):
        if i == 5:
            lines[i] = f"{math.floor(min(listx_n))} {math.ceil(max(listx_n))} xlo xhi\n"
        if i == 6:
            lines[i] = f"{math.floor(min(listy_n))} {math.ceil(max(listy_n))} ylo yhi\n"
        if i == 7:
            lines[i] = f"{math.floor(min(listz_n))} {math.ceil(max(listz_n))} zlo zhi\n"
        if i >= 15:
            lines[i] = "{} {} {} {} {}\n".format(
                listN_int[compt],
                listi_int[compt],
                listx_n[compt],
                listy_n[compt],
                listz_n[compt],
            )
            compt += 1
    fint.close()
    fend = open(file_lmp, "w")
    fend.writelines(lines)
    fend.close()
    return


def write_stl(filename, vertices, face_list):
    """
    Creates an STL file from faces and vertices.

    :param filename: name of the STL file
    :type filename: str
    :param vertices: list of vertices
    :type vertices: array
    :param face_list: list of faces
    :type face_list: array
    """
    with open(filename, "w") as f:
        f.write("solid Created by Gmsh \n")
        for face in face_list:
            p1 = vertices[face[0]]
            p2 = vertices[face[1]]
            p3 = vertices[face[2]]
            OA = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
            OB = [p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2]]
            normal = np.cross(OA, OB)
            f.write(f"facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("  outer loop\n")
            f.write(f"    vertex {p1[0]} {p1[1]} {p1[2]}\n")
            f.write(f"    vertex {p2[0]} {p2[1]} {p2[2]}\n")
            f.write(f"    vertex {p3[0]} {p3[1]} {p3[2]}\n")
            f.write("  endloop\n")
            f.write("endfacet\n")
        f.write("endsolid Created by Gmsh")
    return


def rms_calc(Z):
    """
    Calculates the RMS of a height distribution

    :param Z: height matrix
    :type Z: array

    :return: RMS
    """
    Z = np.asarray(Z).flatten()
    return np.sqrt(np.mean(Z**2))


def random_surf2(m, n, B, xv, yv, sfrM, sfrN, C1, RMS, verbose=True):
    """
    Returns an array with the Z values representing the surface roughness.

    :param m: Wavenumbers
    :type m: array
    :param n: Wavenumbers
    :type n: array
    :param B: The degree of the roughness
    :type B: float
    :param xv: Unique x coordinate values from the objects vertices
    :type xv: array
    :param yv: Unique y coordinate values from the objects vertices
    :type yv: array
    :param sfrN: Vector for the N decomposition
    :type sfrN: array
    :param sfrM: Vector for the M decomposition
    :type sfrM: array
    :param C1: Roughness normalization factor
    :type C1: float
    :param RMS: Root Mean Square
    :type RMS: float
    :param out_pre: Prefix of output files
    :type out_pre: str

    :return: Surface roughness
    """
    if verbose:
        print("====== > Creating random rough surface....")

    Z = 0.0
    for i in range(len(sfrM)):
        for j in range(len(sfrN)):
            if sfrM[i] == 0.0 and sfrN[j] == 0.0:
                continue
            else:
                mod = (sfrM[i] ** 2 + sfrN[j] ** 2) ** (-0.5 * B)
                Z = Z + m[i][j] * mod * np.cos(2 * np.pi * (sfrM[i] * xv + sfrN[j] * yv) + n[i][j])

    if isinstance(C1, str):
        C1 = RMS / rms_calc(Z)

    Z *= C1

    return Z


def make_rough(type_sample, z, nodesurf, vertices, angles):
    """
    Applies roughness on the sample.

    :param type_sample: The type of the sample
    :type type_sample: str
    :param z: Surface roughness to apply on the sample
    :type z: array
    :param nodesurf: List of nodes to be moved
    :type nodesurf: array
    :param vertices: Nodes of the sample
    :type vertices: array
    :param angles: List of angles to be followed by roughness (Only in the case of a faceted wire)
    :type angles: array

    :return: Rough sample
    """
    min_dz = abs(z.min())
    if type_sample == "box" or type_sample == "grain":
        for i in range(len(z)):
            dz = z[i] + min_dz
            node = nodesurf[i]
            index = node[3]
            poss = np.where(vertices[:, 3] == index)
            vertices[poss, 2] = vertices[poss, 2] + dz
    elif type_sample == "wire":
        for i in range(len(z)):
            dz = z[i] + min_dz
            node = nodesurf[i]
            thetaa = node[1]
            index = node[3]
            poss = np.where(vertices[:, 3] == index)
            vertices[poss, 0] = vertices[poss, 0] + dz * np.cos(thetaa)
            vertices[poss, 1] = vertices[poss, 1] + dz * np.sin(thetaa)
    elif type_sample == "poly":
        k = 0
        for i in range(np.shape(z)[0]):
            for j in range(np.shape(z)[1]):
                dz = z[i, j] + min_dz
                node = nodesurf[k]
                index = int(node[3])
                thetaa = np.arctan2(node[1], node[0])
                if thetaa < 0:
                    thetaa = thetaa + 2 * np.pi
                theta_min = np.abs(np.array(angles) - thetaa)
                possi = np.where(abs(theta_min - np.amin(theta_min)) <= 0.01)[0]
                if len(possi) > 1:
                    angle = 0.5 * (angles[int(possi[0])] + angles[int(possi[1])])
                    dz = 0.5 * dz
                elif thetaa == 0:
                    angle = 0
                    dz = 0.5 * dz
                else:
                    angle = angles[int(possi)]
                poss = int(np.where(vertices[:, 3] == index)[0])
                vertices[poss, 0] = vertices[poss, 0] + dz * np.cos(angle)
                vertices[poss, 1] = vertices[poss, 1] + dz * np.sin(angle)
                k += 1
    return vertices


def base(radius, nfaces):
    """
    Creates the base of the faceted wire from radius and number of faces.

    :param radius: Radius of the wire
    :type radius: float
    :param nfaces: Number of faces
    :type nfaces: int

    :return: Points forming the base of the wire and angles associated to each face
    """
    points = []
    theta_list = np.linspace(0, 2 * np.pi, nfaces + 1)
    for i in range(len(theta_list)):
        if theta_list[i] < 0:
            theta_list[i] += 2 * np.pi
    angles = []
    for theta in theta_list[:-1]:
        points.append((radius * np.cos(theta), radius * np.sin(theta)))
    for i in range(len(points)):
        k = i + 1
        if k == len(points):
            k = 0
        x1 = points[int(i)][0]
        x2 = points[int(k)][0]
        y1 = points[int(i)][1]
        y2 = points[int(k)][1]
        the = np.arctan2(0.5 * (y1 + y2), 0.5 * (x1 + x2))
        if the < 0:
            the = the + 2 * np.pi
        angles.append(the)
    angles = np.asarray(angles)
    return points, angles


def cart2cyl(coords: NDArrayDouble) -> NDArrayDouble:
    """
    Calculates cylindrical coordinates from cartesian ones.

    :param matrix: List of cartesian coordinates
    :type matrix: array

    :return: Cylindrical coordinates
    """
    x, y, z, _ = coords.T
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    cyl_matrix = np.column_stack((r, theta, z, coords[:, 3]))
    return cyl_matrix


def cyl2cart(coords: NDArrayDouble) -> NDArrayDouble:
    """
    Calculates cartesian coordinates from cylindrical ones.

    :param matrix: List of cylindrical coordinates
    :type matrix: array

    :return: Cartesian coordinates
    """
    x = coords[:, 0] * np.cos(coords[:, 1])
    y = coords[:, 0] * np.sin(coords[:, 1])
    cart_matrix = np.column_stack((x, y, coords[:, 2], coords[:, 3]))
    return cart_matrix


def faces_normals(obj_points, obj_faces):
    """
    Calculates each face's normals.

    :param obj_points: List of points
    :type obj_points: array
    :param obj_faces: List of faces
    :type obj_faces: array

    :return: List of normals
    """
    list_n = []
    for F in obj_faces:
        A = obj_points[int(F[0] - 1)]
        B = obj_points[int(F[1] - 1)]
        C = obj_points[int(F[2] - 1)]
        centroid = np.array(
            [
                (A[0] + B[0] + C[0]) / 3,
                (A[1] + B[1] + C[1]) / 3,
                (A[2] + B[2] + C[2]) / 3,
            ]
        )
        AB = B - A
        AC = C - A
        n = np.cross(AB, AC)
        n = n / np.linalg.norm(n)
        d = -np.dot(n, A)
        # Check if its the external normal
        point = centroid + n
        if not (does_segment_intersect_plane(point, A, n[0], n[1], n[2], d)):
            n = -1 * n
        list_n.append([n[0], n[1], n[2]])
    return list_n


def does_segment_intersect_plane(S, P, a, b, c, d):
    """
    :param S: Point defining segment with the origin
    :type S: list
    :param P: Plane normal
    :type P: list
    :param a: Plane equation (ax + by + cz + d = 0)
    :type a: float
    :param b: Plane equation (ax + by + cz + d = 0)
    :type b: float
    :param c: Plane equation (ax + by + cz + d = 0)
    :type c: float
    :param d: Plane equation (ax + by + cz + d = 0)
    :type d: float

    :return: Boolean value for plane intersection or not
    """
    x1, y1, z1 = 0.0, 0.0, 0.0
    x2, y2, z2 = S
    p1, p2, p3 = P
    
    denom = a * (x2 - x1) + b * (y2 - y1) + c * (z2 - z1)
    if denom == 0.0 and a * x2 + b * y2 + c * z2 + d == 0:
        return True
    else:
        t = (a * (p1 - x1) + b * (p2 - y1) + c * (p3 - z1)) / denom
        if (t >= 0) and (t <= 1):
            return True

    return False


def identify_edge_and_corner_nodes(nodesurf):
    """
    From surface nodes, finds all nodes located on edges and corners.

    :param nodesurf: List of surface nodes
    :type nodesurf: array

    :return: List of nodes located on edges and list of nodes located on corners
    """
    all_points = np.concatenate(nodesurf)
    list_index = all_points[:, 3]
    
    unique, counts = np.unique(list_index, return_counts=True)
    node_edge = unique[counts == 2]
    node_corner = unique[counts >= 3]
    
    return node_edge, node_corner


def make_rough_wulff(vertices, B, C1, RMS, N, M, nodesurf, node_edge, node_corner, list_n):
    """
    Applies roughness on the sample in the case of a Wulff Shaped NP.

    :param vertices: Nodes of the sample
    :type vertices: array
    :param B: The degree of the roughness
    :type B: float
    :param C1: Roughness normalization factor
    :type C1: float
    :param RMS: Root Mean Square
    :type RMS: float
    :param N: Scaling cartesian position
    :type N: int
    :param M: Scaling cartesian position
    :type M: int
    :param nodesurf: List of surface nodes
    :type nodesurf: array
    :param node_edge: List of nodes located on edges
    :type node_edge: array
    :param node_corner: List of nodes located on corners
    :type node_corner: array
    :param list_n: List of face's normals
    :type list_n: list

    :return: Rough Wulff sample
    """
    sfrN, sfrM = np.linspace(-N, N, 2 * N + 1), np.linspace(-M, M, 2 * M + 1)
    for k in range(len(nodesurf)):
        if k == 0:
            print(
                f"====== > Creating random rough surface n° {k + 1}, ",
                end=" ",
                flush=True,
            )
        elif k == len(nodesurf) - 1:
            print(f"{k + 1}.")
        else:
            print(f"{k + 1}, ", end=" ", flush=True)
        surf = np.array(nodesurf[k])
        n1 = np.array(list_n[k])
        surf_rot = align_surface_normal_with_z(surf, n1)
        surf_norm = normalize(surf_rot)
        xv = surf_norm[:, 0]
        yv = surf_norm[:, 1]
        m, n = random_numbers(sfrN, sfrM)

        z = random_surf2(m, n, B, xv, yv, sfrM, sfrN, C1, RMS, verbose=False)

        z = z + abs(np.min(z))
        for i in range(len(surf)):
            p = surf[i]
            index = p[3]
            if index in node_edge:
                delta_z = z[i] / 2
            elif index in node_corner:
                delta_z = z[i] / 3
            else:
                delta_z = z[i]
            poss = np.where(vertices[:, 3] == index)
            vertices[poss, 0] = vertices[poss, 0] + delta_z * n1[0]
            vertices[poss, 1] = vertices[poss, 1] + delta_z * n1[1]
            vertices[poss, 2] = vertices[poss, 2] + delta_z * n1[2]
    return vertices


def align_surface_normal_with_z(surf, n1):
    """
    Rotates a surface oriented along n1 axis in order to be oriented along z-axis.

    :param surf: List of nodes from the surface
    :type surf: array
    :param n1: Surface normal
    :type n1: array

    :return: Rotated surface
    """
    n2 = np.array([0, 0, 1])
    n = np.cross(n1, n2)
    
    if np.all(n == 0):
        n = n2
        
    n = n / np.linalg.norm(n)
    theta = np.arccos(np.dot(n1, n2))
    rot = R.from_rotvec(theta * np.array(n))
    
    surf_rot = np.array([np.append(rot.apply(p[:3]), p[3]) for p in surf])
    
    return surf_rot


def normalize(surf):
    """
    Normalizes the coordinates of points composing the surface.

    :param surf: List of nodes of the surface
    :type surf: array

    :return: Normalized surface
    """
    X, Y, Z, T = surf.T
    Xf = (X / np.max(np.abs(X)) + 1) / 2
    Yf = (Y / np.max(np.abs(Y)) + 1) / 2
    Zf = np.zeros_like(Z)
    return np.column_stack((Xf, Yf, Zf, T))


def cube_faces(length):
    """
    Generates the points and faces of a cube

    :param length: Size of the cube
    :type length: float

    :return: Points and faces of the cube
    """
    vertex_coords = np.array([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [length, length, 0],
        [0, 0, length],
        [length, 0, length],
        [0, length, length],
        [length, length, length],
    ])
    face_indices = np.array([
        [1, 2, 3, 4],
        [1, 2, 5, 6],
        [2, 4, 6, 8],
        [4, 8, 3, 7],
        [1, 5, 3, 7],
        [5, 6, 7, 8],
    ])
    return vertex_coords, face_indices


def rotate_obj_wulff(obj_points, orien_x, orien_z):
    """
    :param obj_points: Points describing the facets of the wulff-shape
    :type obj_points: array
    :param orien_x: Orientation along x-axis
    :type orien_x: list
    :param orien_z: Orientation along z-axis
    :type orien_z: list

    :return: Points respecting the desired orientation
    """
    n2 = np.array([0, 0, 1])
    n = np.cross(orien_z, n2)
    if n[0] == 0 and n[1] == 0 and n[2] == 0:
        n = n2
    n = n / np.linalg.norm(n)
    theta = np.arccos(np.dot(orien_z, n2) / (np.linalg.norm(orien_z) * np.linalg.norm(n2)))
    rot = R.from_rotvec(theta * np.array(n))
    surf_rot = []
    for p in obj_points:
        point_rot = rot.apply(p)
        surf_rot.append([point_rot[0], point_rot[1], point_rot[2]])
    surf_rot = np.asarray(surf_rot)
    x_rot = rot.apply([1, 0, 0])
    theta_x = np.arccos(np.dot(x_rot, orien_x) / (np.linalg.norm(x_rot) * np.linalg.norm(orien_x)))
    R_x = np.array(
        [
            [np.cos(theta_x), -1 * np.sin(theta_x), 0],
            [np.sin(theta_x), np.cos(theta_x), 0],
            [0, 0, 1],
        ]
    )
    points_f = []
    for p in surf_rot:
        rot_f = np.dot(R_x, p)
        points_f.append([rot_f[0], rot_f[1], rot_f[2]])
    obj_points_f = np.asarray(points_f)
    return obj_points_f


def refine_3Dmesh(type_sample, out_pre, ns, alpha, ext_fem) -> None:
    """

    :param type_sample: Type of object
    :type type_sample: str
    :param out_pre: Outfit file name
    :type out_pre: str
    :param ns: Mesh size
    :type ns: float
    :param alpha: Refine mesh factor
    :type alpha: float
    :param ext_fem: FEM extensions list
    :type ext_fem: list

    :return: Refined 3D meshs for all required formats
    """
    print(f"====== > Refining mesh for {type_sample} object")

    if "stl" in ext_fem:
        ext_fem.remove("stl")

    refinement_strategy_map = {
        "box": refine_box,
        "grain": refine_box,
        "wire": refine_wire,
        "poly": refine_wire,
        "sphere": refine_sphere,
        "wulff": refine_sphere,
        "cube": refine_sphere,
    }
    refinement_angle_map = {
        "box": 45.0,
        "grain": 45.0,
        "wire": 0.0,
        "poly": 0.0,
        "sphere": 0.0,
        "wulff": 45.0,
        "cube": 45.0,
    }

    refine = refinement_strategy_map[type_sample]
    angle = refinement_angle_map[type_sample]

    refine(out_pre, ns, alpha, angle, ext_fem)


def refine_box(out_pre, ns, alpha, angle, ext_fem):
    """

    :param out_pre: Outfit file name
    :type out_pre: str
    :param ns: Mesh size
    :type ns: float
    :param alpha: Refine mesh factor
    :type alpha: float
    :param angle: Angle value for facets detection
    :type angle: float
    :param ext_fem: FEM extensions list
    :type ext_fem: list

    :return: Refined box mesh
    """
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    # Let's merge an STL mesh that we would like to remesh.
    gmsh.merge(out_pre + ".stl")

    # Force curves to be split on given angle:
    gmsh.model.mesh.classifySurfaces(np.deg2rad(angle), True, False)

    gmsh.model.mesh.createGeometry()
    s = gmsh.model.getEntities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])
    gmsh.model.geo.synchronize()

    # Extract node information
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    # Reshape the node coordinates into a more user-friendly format
    vertices = node_coords.reshape(-1, 3)
    # Variables
    z_max = np.max(vertices.T[2])
    z_min = np.min(vertices.T[2])
    formula = f"{ns} + ({ns} - {alpha * ns})/({z_min} - {z_max}) * (z - {z_min})"
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", formula)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    gmsh.model.mesh.generate(3)
    gmsh.write(out_pre + ".stl")
    for e in ext_fem:
        gmsh.write(out_pre + "." + e)


def refine_wire(out_pre, ns, alpha, angle, ext_fem):
    """

    :param out_pre: Outfit file name
    :type out_pre: str
    :param ns: Mesh size
    :type ns: float
    :param alpha: Refine mesh factor
    :type alpha: float
    :param angle: Angle value for facets detection
    :type angle: float
    :param ext_fem: FEM extensions list
    :type ext_fem: list

    :return: Refined wire mesh
    """
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    # Let's merge an STL mesh that we would like to remesh.
    gmsh.merge(out_pre + ".stl")
    # Force curves to be split on given angle:
    gmsh.model.mesh.classifySurfaces(np.deg2rad(angle), True, False)

    gmsh.model.mesh.createGeometry()

    s = gmsh.model.getEntities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])
    gmsh.model.geo.synchronize()

    # Extract node information
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    # Reshape the node coordinates into a more user-friendly format
    vertices = node_coords.reshape(-1, 3)

    r_max = np.max(np.power(np.power(vertices.T[0], 2) + np.power(vertices.T[1], 2), 0.5))

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    formula = f"abs((({alpha * ns}-{ns})/{r_max})*sqrt(x*x + y*y) + {ns})"
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", formula)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(out_pre + ".stl")
    for e in ext_fem:
        gmsh.write(out_pre + "." + e)


def refine_sphere(out_pre, ns, alpha, angle, ext_fem):
    """

    :param out_pre: Outfit file name
    :type out_pre: str
    :param ns: Mesh size
    :type ns: float
    :param alpha: Refine mesh factor
    :type alpha: float
    :param angle: Angle value for facets detection
    :type angle: float
    :param ext_fem: FEM extensions list
    :type ext_fem: list

    :return: Refined sphere mesh
    """
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    # Let's merge an STL mesh that we would like to remesh.
    gmsh.merge(out_pre + ".stl")

    # Force curves to be split on given angle:
    gmsh.model.mesh.classifySurfaces(np.deg2rad(angle), True, False)

    gmsh.model.mesh.createGeometry()

    s = gmsh.model.getEntities(2)
    loop_tag = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([loop_tag])
    gmsh.model.geo.synchronize()

    # Extract node information
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    # Reshape the node coordinates into a more user-friendly format
    vertices = node_coords.reshape(-1, 3)

    r_max = np.max(
        np.power(
            np.power(vertices.T[0], 2) + np.power(vertices.T[1], 2) + np.power(vertices.T[2], 2),
            0.5,
        )
    )

    formula = f"abs((({alpha * ns}-{ns})/{r_max})*sqrt(x*x + y*y + z*z) + {ns})"
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", formula)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(out_pre + ".stl")
    for e in ext_fem:
        gmsh.write(out_pre + "." + e)
