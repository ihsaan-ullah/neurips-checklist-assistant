# ------------------------------------------
# Imports
# ------------------------------------------
from result_compiler import ResultCompiler


if __name__ == "__main__":
    print("############################################")
    print("### Starting Result Compilation")
    print("############################################\n")

    # Init scoring
    scoring = ResultCompiler(result_dir="./Assistant_Result")

    # Set directories
    scoring.set_directories()

    # Start timer
    scoring.start_timer()

    # Load ingestion result
    scoring.load_ingestion_result()

    # Write detailed results
    scoring.write_detailed_results()

    # Stop timer
    scoring.stop_timer()

    # Show duration
    scoring.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Result compilation successful!")
    print("----------------------------------------------\n\n")