# ---------------------------------------------------------------------------
# Title: Main code pyrough
# Authors: Jonathan Amodeo, Javier Gonzalez, Jennifer Izaguirre, Christophe Le Bourlot, Hugo Iteney
# Date: June 01, 2022
#
# Pyrough is a code used to provide the user with either FEM or MD files of various objects that
# have had random surface roughness applied to it. The objects included in this code are nanowire,
# slab, sphere. In the main code the user must provide what type of object thy desired. Based on
# this the code will read the parameters from the json file provided and return an stl if FEM is
# desired or lmp file if MD is desired.
# ---------------------------------------------------------------------------
import argparse
import subprocess
from pathlib import Path

from src.Param_class import Parameter
from src.Sample_class import Sample, make_atom_grain, make_box


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pyrough: Generate FEM or MD files with random surface roughness."
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to the json configuration file. Not required if -surface is provided.",
    )
    parser.add_argument(
        "-surface",
        nargs=4,
        metavar=("path", "size", "zmin", "zmax"),
        help="Surface analysis arguments.",
    )
    args = parser.parse_args()

    if not args.config and not args.surface:
        parser.error("Either config or -surface arguments must be provided.")

    if args.config and args.surface:
        parser.error("Only one of config or -surface can be provided.")

    return args


def main() -> None:
    print("#####################################################################################")
    print("#                                      Pyrough                                      #")
    print("#                         Jonathan Amodeo & Hugo Iteney 2023                        #")
    print("#####################################################################################")

    args = parse_arguments()

    out_pre = Path(args.config).stem
    param = Parameter(args.config)

    if args.surface:
        current_dir = Path(__file__).resolve().parent
        exe_path = current_dir / "src/Surface_Analysis.py"

        if not exe_path.exists():
            print(f"Error: {exe_path} not found.")
            return

        subprocess.call(["python", exe_path, *args.surface])
        return

    if param.type_S == "grain":
        vertices, FEM_stl = make_box(
            param.type_S,
            2 * (1 + param.eta),
            param.C1,
            param.RMS,
            param.N,
            param.M,
            param.length,
            param.height,
            param.width,
            param.ns,
            param.alpha,
            param.raw_stl,
            out_pre,
            param.ext_fem,
        )

        make_atom_grain(
            FEM_stl,
            param.lattice_structure1,
            param.lattice_parameter1,
            param.material1,
            param.orien_x1,
            param.orien_y1,
            param.orien_z1,
            param.lattice_structure2,
            param.lattice_parameter2,
            param.material2,
            param.orien_x2,
            param.orien_y2,
            param.orien_z2,
            vertices,
            out_pre,
            param.ext_ato,
        )
        print("JOB DONE!" + "  File name: " + out_pre + ".lmp")
        return

    sample = Sample(param.type_S)

    vertices, FEM_stl = sample.make_stl(
        param.type_S,
        param.eta,
        param.C1,
        param.RMS,
        param.N,
        param.M,
        param.radius,
        param.length,
        param.height,
        param.width,
        param.ns,
        param.alpha,
        param.raw_stl,
        param.nfaces,
        param.surfaces,
        param.energies,
        param.n_at,
        param.lattice_structure,
        param.lattice_parameter,
        param.material,
        param.orien_x,
        param.orien_z,
        out_pre,
        param.ext_fem,
    )
    print("====== > FEM JOB DONE !")

    for ext in param.ext_fem:
        print("====== > File name: " + out_pre + "." + str(ext))

    if "stl" not in param.ext_fem:
        print("====== > File name: " + out_pre + ".stl")

    if param.output(args.config) == "ATOM":
        sample.make_atom(
            FEM_stl,
            param.lattice_structure,
            param.lattice_parameter,
            param.material,
            param.orien_x,
            param.orien_y,
            param.orien_z,
            vertices,
            out_pre,
            param.ext_ato,
        )
        print("====== > ATOM JOB DONE !")

        # Print the name of the files generated
        for ext in param.ext_ato:
            print("====== > File name: " + out_pre + "." + str(ext))


if __name__ == "__main__":
    main()
