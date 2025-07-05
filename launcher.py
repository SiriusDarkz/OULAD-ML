import sys, os
sys.path.insert(0, os.path.abspath("src"))

from oulad.etl import transform

if __name__ == "__main__":
    transform.main()