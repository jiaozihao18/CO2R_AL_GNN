import argparse
from ase.db import connect
import ase.io

def main():
    parser = argparse.ArgumentParser(description='Write VASP output to ASE database.')
    parser.add_argument('db_path', type=str, help='Path to the database file.')
    parser.add_argument('atoms_path', type=str, help='Path to the VASP OUTCAR file.')
    args = parser.parse_args()
    try:
        db = connect(args.db_path)
        atoms = ase.io.read(args.atoms_path, index=-1)
        db.write(atoms, info=args.atoms_path)
        print("Data written to database successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
