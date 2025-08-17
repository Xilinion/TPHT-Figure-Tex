#!/usr/bin/env python3
"""
Script to calculate and extract in-text data from CSV files for the paper.

This script reads all CSV files from ../csv directory and provides a framework
to calculate specific statistics and values mentioned in the paper text.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import glob

class DataProcessor:
    def __init__(self):
        """Initialize the data processor and load all CSV files."""
        self.csv_dir = Path("../csv")
        self.output_file = "./in_text_data.csv"
        self.latex_file = "./in_text_data.tex"
        
        # Dictionary to store all loaded dataframes
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
        # List to store tuples for output CSV
        self.results: List[Tuple[int, str, Any]] = []
        
        # Load all CSV files
        self.load_csv_files()
        
        # Counter for generating unique IDs
        self.id_counter = 1
    
    def load_csv_files(self):
        """Load all CSV files from the csv directory into dataframes."""
        csv_files = glob.glob(str(self.csv_dir / "*.csv"))
        
        for csv_file in csv_files:
            # Extract filename without extension to use as key
            filename = Path(csv_file).stem
            
            try:
                df = pd.read_csv(csv_file)
                self.dataframes[filename] = df
                print(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"  Columns: {list(df.columns)}")
                print()
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    def add_result(self, name: str, value: Any) -> None:
        """Add a result tuple to the results list."""
        self.results.append((self.id_counter, name, value))
        self.id_counter += 1
    
    def get_dataframe(self, name: str) -> pd.DataFrame:
        """Get a dataframe by name with error checking."""
        if name not in self.dataframes:
            available = ', '.join(self.dataframes.keys())
            raise ValueError(f"Dataframe '{name}' not found. Available: {available}")
        return self.dataframes[name]
    
    # ==========================================================================
    # CALCULATION FUNCTIONS
    # Add your calculation functions below. Each function should call add_result()
    # ==========================================================================
    
    def calculate_max_ycsb_throughput(self):
        """
        Example function: Calculate the maximum throughput in YCSB results.
        
        Uses: ycsb_results.csv
        Calculates: Maximum run throughput across all experiments
        """
        df = self.get_dataframe('ycsb_results')
        
        # Find maximum run throughput
        max_throughput = df['run_throughput (ops/s)'].max()
        max_throughput_millions = max_throughput / 1_000_000
        
        # Find which object achieved this maximum
        max_row = df.loc[df['run_throughput (ops/s)'].idxmax()]
        object_id = max_row['object_id']
        case_id = max_row['case_id']
        
        self.add_result("max_ycsb_run_throughput_ops", max_throughput)
        self.add_result("max_ycsb_run_throughput_millions", round(max_throughput_millions, 1))
        self.add_result("max_ycsb_object_id", object_id)
        self.add_result("max_ycsb_case_id", case_id)
    
    def calculate_data_size_scaling_performance(self):
        """
        Example function: Calculate performance metrics from data size scaling.
        
        Uses: data_size_scaling_results.csv
        Calculates: Performance comparison between smallest and largest table sizes
        """
        df = self.get_dataframe('data_size_scaling_results')
        
        # Get performance at smallest and largest table sizes for TPHT (object_id=17)
        tpht_data = df[df['object_id'] == 17]
        
        if not tpht_data.empty:
            min_size = tpht_data['table_size'].min()
            max_size = tpht_data['table_size'].max()
            
            min_throughput = tpht_data[tpht_data['table_size'] == min_size]['throughput (ops/s)'].iloc[0]
            max_throughput = tpht_data[tpht_data['table_size'] == max_size]['throughput (ops/s)'].iloc[0]
            
            # Calculate performance ratio
            performance_ratio = min_throughput / max_throughput
            
            self.add_result("tpht_min_table_size", min_size)
            self.add_result("tpht_max_table_size", max_size)
            self.add_result("tpht_min_size_throughput", min_throughput)
            self.add_result("tpht_max_size_throughput", max_throughput)
            self.add_result("tpht_size_performance_ratio", round(performance_ratio, 2))
    
    def calculate_occupancy_statistics(self):
        """
        Example function: Calculate occupancy statistics.
        
        Uses: occupancy_experimental_box.csv
        Calculates: Statistics about occupancy distribution
        """
        df = self.get_dataframe('occupancy_experimental_box')
        
        # Calculate total probability for occupancies 0-2
        low_occupancy_prob = df[df['occupancy'] <= 2]['median'].sum()
        
        # Find median occupancy (where cumulative probability >= 0.5)
        df_sorted = df.sort_values('occupancy')
        cumulative_prob = df_sorted['median'].cumsum()
        median_occupancy = df_sorted[cumulative_prob >= 0.5]['occupancy'].iloc[0]
        
        self.add_result("occupancy_0_2_probability", round(low_occupancy_prob, 4))
        self.add_result("median_occupancy_level", median_occupancy)
    
    # ==========================================================================
    # ADD YOUR CALCULATION FUNCTIONS HERE
    # ==========================================================================
    
    def your_calculation_function_template(self):
        """
        Template for adding new calculation functions.
        
        Uses: [specify which CSV file(s)]
        Calculates: [describe what this function calculates]
        """
        # Example:
        # df = self.get_dataframe('your_csv_name')
        # result = df['column'].mean()  # or any other calculation
        # self.add_result("your_metric_name", result)
        pass
    
    # ==========================================================================
    # MAIN EXECUTION FUNCTIONS
    # ==========================================================================
    
    def run_all_calculations(self):
        """Run all calculation functions."""
        print("Running all calculations...")
        
        # Add calls to all your calculation functions here
        self.calculate_max_ycsb_throughput()
        self.calculate_data_size_scaling_performance()
        self.calculate_occupancy_statistics()
        
        # Add calls to your new functions here:
        # self.your_new_function()
        
        print(f"Generated {len(self.results)} results")
    
    def make_latex_safe_name(self, name: str) -> str:
        """Convert a name to be LaTeX-safe for use in newcommand."""
        # Replace invalid characters with valid ones
        latex_name = name.replace("_", "")
        latex_name = latex_name.replace("-", "")
        latex_name = latex_name.replace(".", "")
        latex_name = latex_name.replace(" ", "")
        
        # Ensure it starts with a letter
        if latex_name and not latex_name[0].isalpha():
            latex_name = "data" + latex_name
        
        return latex_name
    
    def save_results(self):
        """Save results to CSV file."""
        if not self.results:
            print("No results to save!")
            return
        
        # Create output dataframe
        results_df = pd.DataFrame(self.results, columns=['id', 'name', 'value'])
        
        # Save to CSV
        results_df.to_csv(self.output_file, index=False)
        print(f"Results saved to {self.output_file}")
        
        # Display results for verification
        print("\nGenerated results:")
        print(results_df.to_string(index=False))
    
    def save_latex_commands(self):
        """Save results as LaTeX newcommand definitions."""
        if not self.results:
            print("No results to save as LaTeX!")
            return
        
        latex_lines = []
        
        for _, name, value in self.results:
            # Create LaTeX-safe command name
            latex_name = self.make_latex_safe_name(name)
            
            # Format value appropriately
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    # Format floats with appropriate precision
                    if value.is_integer():
                        formatted_value = str(int(value))
                    else:
                        formatted_value = f"{value:.10g}"  # Use general format to avoid unnecessary decimals
                else:
                    formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            # Create newcommand line
            latex_line = f"\\newcommand{{\\{latex_name}}}{{{formatted_value}}}"
            latex_lines.append(latex_line)
        
        # Join with empty lines between commands
        latex_content = "\n\n".join(latex_lines)
        
        # Save to file
        with open(self.latex_file, 'w') as f:
            f.write(latex_content)
        
        print(f"LaTeX commands saved to {self.latex_file}")
        print("\nGenerated LaTeX commands:")
        print(latex_content)
    
    def show_available_data(self):
        """Display information about available datasets."""
        print("Available datasets:")
        print("=" * 50)
        for name, df in self.dataframes.items():
            print(f"{name}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            if not df.empty:
                print(f"  Sample data:")
                print(f"    {df.head(2).to_string(index=False)}")
            print()

def main():
    """Main execution function."""
    processor = DataProcessor()
    
    # Show available data for reference
    processor.show_available_data()
    
    # Run all calculations
    processor.run_all_calculations()
    
    # Save results
    processor.save_results()
    
    # Save LaTeX commands
    processor.save_latex_commands()

if __name__ == "__main__":
    main()