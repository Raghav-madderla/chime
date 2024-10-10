"""Command line interface."""

import os
import sys
from .model.parameters import Parameters
from .model.sir import Sir
from penn_chime.model.ml_disease_prediction import MLDiseasePrediction

def run(argv):
    """Eun cli."""
    p = Parameters.create(os.environ, argv[1:])
    m = Sir(p)

    for df, name in (
        (m.sim_sir_w_date_df, "sim_sir_w_date"),
        (m.admits_df, "projected_admits"),
        (m.census_df, "projected_census"),
        (m.ppe_df, 'ppe_data')
    ):
        df.to_csv(f"{p.current_date}_{name}.csv")



def run_ml(argv):
    """Run the machine learning prediction model."""
    print("ML model is running...")
    try:
        predictor = MLDiseasePrediction()
        predictor.run()  
        print("ML model finished running.") 
    except Exception as e:
        print(f"Error running ML model: {e}")

def main():
    """Main entry point."""
    print("Main function is called.") 
    if len(sys.argv) > 1 and sys.argv[1] == "ml":
        print("Running ML model from CLI...")
        run_ml(sys.argv)
    else:
        print("Running SIR model from CLI...")  
        run(sys.argv)

if __name__ == "__main__":
    main()
