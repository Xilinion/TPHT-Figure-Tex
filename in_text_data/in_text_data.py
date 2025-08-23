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
    
    def calculate_throughput_metrics(self):
        """
        Calculate throughput metrics for the throughput.tex section.
        
        Uses: ycsb_results.csv
        Calculates: Various throughput statistics using hash table macros
        """
        df = self.get_dataframe('ycsb_results')
        
        # Filter for entry_id=0 only as specified
        df_filtered = df[df['entry_id'] == 0].copy()
        
        # Object ID to macro mapping
        object_to_macro = {
            6: 'htthree',   # Cuckoo
            7: 'htfour',    # Iceberg
            15: 'htfive',   # Junction
            17: 'htone',    # TPHT
            20: 'httwo'     # Blast
        }
        
        # === \httwo (Blast, object_id=20) metrics ===
        httwo_data = df_filtered[df_filtered['object_id'] == 20]
        
        # Average fill throughput for Load phase (case_id=17) - keep raw values
        httwo_load_fill_raw = httwo_data[httwo_data['case_id'] == 17]['fill_throughput (ops/s)'].values[0]
        self.add_result("httwo_avg_fill_throughput", round(httwo_load_fill_raw / 1_000_000, 1))
        
        # Average run throughput across all run phases (case_ids 17-22) - keep raw values
        httwo_run_throughputs_raw = httwo_data[httwo_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
        httwo_avg_run_raw = httwo_run_throughputs_raw.mean()
        self.add_result("httwo_avg_run_throughput", round(httwo_avg_run_raw / 1_000_000, 1))
        
        # === \htone (TPHT, object_id=17) metrics ===
        htone_data = df_filtered[df_filtered['object_id'] == 17]
        
        # Average fill throughput for Load phase - keep raw values
        htone_load_fill_raw = htone_data[htone_data['case_id'] == 17]['fill_throughput (ops/s)'].values[0]
        self.add_result("htone_avg_fill_throughput", round(htone_load_fill_raw / 1_000_000, 1))
        
        # Average run throughput across all run phases - keep raw values
        htone_run_throughputs_raw = htone_data[htone_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
        htone_avg_run_raw = htone_run_throughputs_raw.mean()
        self.add_result("htone_avg_run_throughput", round(htone_avg_run_raw / 1_000_000, 1))
        
        # === Speedup calculations ===
        # Find fastest baseline for comparison (excluding httwo and htone)
        baseline_objects = [6, 7, 15]  # Cuckoo, Iceberg, Junction
        
        # Fill throughput comparison - use raw values
        baseline_fill_max_raw = df_filtered[(df_filtered['object_id'].isin(baseline_objects)) & 
                                           (df_filtered['case_id'] == 17)]['fill_throughput (ops/s)'].max()
        
        httwo_fill_speedup = httwo_load_fill_raw / baseline_fill_max_raw
        htone_fill_speedup = htone_load_fill_raw / baseline_fill_max_raw
        
        self.add_result("httwo_fill_speedup", round(httwo_fill_speedup, 2))
        self.add_result("htone_fill_speedup_percent", round(htone_fill_speedup * 100, 1))
        
        # Run throughput comparison - calculate fastest avg run throughput among baselines
        baseline_avg_runs = []
        for obj_id in baseline_objects:
            baseline_obj_data = df_filtered[df_filtered['object_id'] == obj_id]
            baseline_run_throughputs = baseline_obj_data[baseline_obj_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
            baseline_avg_run = baseline_run_throughputs.mean()
            baseline_avg_runs.append(baseline_avg_run)
        
        baseline_run_max_raw = max(baseline_avg_runs)
        
        httwo_run_speedup = httwo_avg_run_raw / baseline_run_max_raw
        htone_run_speedup = htone_avg_run_raw / baseline_run_max_raw
        
        self.add_result("httwo_run_speedup", round(httwo_run_speedup, 2))
        self.add_result("htone_run_speedup_percent", round(htone_run_speedup*100, 1))

        self.add_result("httwo_speedup_percent", round((httwo_fill_speedup + httwo_run_speedup) * 100 / 2 - 100, 1))
        
        # === Workload-specific analysis ===
        # Run A to Run B comparison (case 17->18)
        for obj_id, macro in object_to_macro.items():
            obj_data = df_filtered[df_filtered['object_id'] == obj_id]
            
            run_a_throughput_raw = obj_data[obj_data['case_id'] == 17]['run_throughput (ops/s)'].values[0]
            run_b_throughput_raw = obj_data[obj_data['case_id'] == 18]['run_throughput (ops/s)'].values[0]
            
            # Calculate percentage change using raw values
            if run_a_throughput_raw > 0:
                change_percent = ((run_b_throughput_raw - run_a_throughput_raw) / run_a_throughput_raw) * 100
                speedup_factor = run_b_throughput_raw / run_a_throughput_raw
                
                self.add_result(f"{macro}_a_to_b_speedup", round(speedup_factor, 1))
                self.add_result(f"{macro}_a_to_b_change_percent", round(abs(change_percent), 1))
        
        # === Positive vs Negative query analysis ===
        # Run C vs Run C- comparison (case 19 vs 22)
        for obj_id, macro in object_to_macro.items():
            obj_data = df_filtered[df_filtered['object_id'] == obj_id]
            
            run_c_pos_raw = obj_data[obj_data['case_id'] == 19]['run_throughput (ops/s)'].values[0]  # Positive
            run_c_neg_raw = obj_data[obj_data['case_id'] == 22]['run_throughput (ops/s)'].values[0]  # Negative
            
            # Calculate which is higher and by how much using raw values
            if run_c_neg_raw > run_c_pos_raw:
                speedup = run_c_neg_raw / run_c_pos_raw
                self.add_result(f"{macro}_neg_over_pos_speedup", round(speedup, 1))
            else:
                speedup = run_c_pos_raw / run_c_neg_raw
                self.add_result(f"{macro}_pos_over_neg_speedup", round(speedup, 1))
            
            # Also calculate percentage difference using raw values
            diff_percent = abs((run_c_neg_raw - run_c_pos_raw) / max(run_c_pos_raw, run_c_neg_raw)) * 100
            self.add_result(f"{macro}_pos_neg_diff_percent", round(diff_percent, 1))
        
        # === Junction's Run C speedup over Load ===
        junction_data = df_filtered[df_filtered['object_id'] == 15]
        junction_load_fill_raw = junction_data[junction_data['case_id'] == 17]['fill_throughput (ops/s)'].values[0]
        junction_run_c_raw = junction_data[junction_data['case_id'] == 19]['run_throughput (ops/s)'].values[0]
        
        junction_c_over_load_speedup = junction_run_c_raw / junction_load_fill_raw
        self.add_result("htfive_run_c_over_load_speedup", round(junction_c_over_load_speedup, 1))
        
        # === htfour workload variance analysis ===
        # Calculate coefficient of variation (CV) for htfour across all run workloads
        htfour_data = df_filtered[df_filtered['object_id'] == 7]
        htfour_run_throughputs = htfour_data[htfour_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
        htfour_mean = htfour_run_throughputs.mean()
        htfour_std = htfour_run_throughputs.std()
        htfour_cv_percent = (htfour_std / htfour_mean) * 100
        self.add_result("htfour_workload_variance_percent", round(htfour_cv_percent, 1))
    
    def calculate_tradeoff_metrics(self):
        """
        Calculate speed/space tradeoff metrics for the tradeoff.tex section.
        
        Uses: throughput_space_eff_results.csv
        Calculates: Space efficiency metrics and throughput analysis
        """
        df = self.get_dataframe('throughput_space_eff_results')
        
        # Object ID to macro mapping
        object_to_macro = {
            6: 'htthree',   # Cuckoo
            7: 'htfour',    # Iceberg
            15: 'htfive',   # Junction
            17: 'htone',    # TPHT
            20: 'httwo'     # Blast
        }

        baseline_objects = [6, 7, 15]  # Cuckoo, Iceberg, Junction

        htone_space_eff_percent = 0
        httwo_space_eff_percent = 0

        baseline_space_eff_percent = []
        
        # === Maximum space efficiency analysis ===
        for obj_id, macro in object_to_macro.items():
            obj_data = df[df['object_id'] == obj_id]
            max_space_eff = obj_data['space_efficiency'].max()
            # Convert to percentage and round
            max_space_eff_percent = max_space_eff * 100

            if obj_id == 17:
                htone_space_eff_percent = max_space_eff_percent
            if obj_id == 20:
                httwo_space_eff_percent = max_space_eff_percent

            if obj_id in baseline_objects:
                baseline_space_eff_percent.append(max_space_eff_percent)

            self.add_result(f"{macro}_max_space_efficiency_percent", round(max_space_eff_percent, 1))
        
        mem_shave_percent = []

        for p in baseline_space_eff_percent:
            mem_shave_percent.append((1 - p / htone_space_eff_percent)*100)

        max_space_eff_percent = max(mem_shave_percent)
        self.add_result("htone_mem_shave_upper_percent", round(max_space_eff_percent, 1))

        min_space_eff_percent = min(mem_shave_percent)
        self.add_result("htone_mem_shave_lower_percent", round(min_space_eff_percent, 1))

        mean_space_eff_percent = np.mean(mem_shave_percent)
        self.add_result("htone_mem_shave_mean_percent", round(mean_space_eff_percent, 1))

        mem_shave_percent_httwo = []
        for p in baseline_space_eff_percent:
            mem_shave_percent_httwo.append((1 - p / httwo_space_eff_percent)*100)
        
        mean_space_eff_percent_httwo = np.mean(mem_shave_percent_httwo)
        self.add_result("httwo_mem_shave_mean_percent", round(mean_space_eff_percent_httwo, 1))




        # === Throughput drop analysis from low to high load factor ===
        # For positive query (case_id=9), analyze throughput drop from 0.05 to 0.7 load factor
        case9_data = df[df['case_id'] == 9]
        
        for obj_id, macro in object_to_macro.items():
            obj_case9 = case9_data[case9_data['object_id'] == obj_id]
            
            # Get throughput at 0.05 and 0.7 load factors
            low_lf_data = obj_case9[obj_case9['load_factor'] == 0.05]
            high_lf_data = obj_case9[obj_case9['load_factor'] == 0.7]
            
            if not low_lf_data.empty and not high_lf_data.empty:
                low_throughput = low_lf_data['throughput_millions'].iloc[0]
                high_throughput = high_lf_data['throughput_millions'].iloc[0]
                
                # Calculate percentage drop
                drop_percent = ((low_throughput - high_throughput) / low_throughput) * 100
                self.add_result(f"{macro}_throughput_drop_percent", round(drop_percent, 1))
            
            # Special case for htfour speedup near 0.5 load factor
            if obj_id == 7:  # htfour
                mid_lf_data_1 = obj_case9[obj_case9['load_factor'] == 0.55]
                mid_lf_data_2 = obj_case9[obj_case9['load_factor'] == 0.6]
                if not mid_lf_data_1.empty and not mid_lf_data_2.empty:
                    speedup_percent = ((mid_lf_data_2['throughput_millions'].iloc[0] - mid_lf_data_1['throughput_millions'].iloc[0]) / mid_lf_data_1['throughput_millions'].iloc[0]) * 100
                    self.add_result(f"{macro}_mid_load_speedup_percent", round(speedup_percent, 1))
        
        # === Performance comparison at 0.05 load factor ===
        # htfive vs httwo at 0.05 load factor for positive query (case_id=9)
        case9_data = df[df['case_id'] == 9]
        low_lf_case9 = case9_data[case9_data['load_factor'] == 0.05]
        
        htfive_low_throughput = low_lf_case9[low_lf_case9['object_id'] == 15]['throughput_millions'].iloc[0]
        httwo_low_throughput = low_lf_case9[low_lf_case9['object_id'] == 20]['throughput_millions'].iloc[0]
        
        throughput_ratio = htfive_low_throughput / httwo_low_throughput
        self.add_result("htfive_over_httwo_low_load_ratio", round(throughput_ratio, 1))
    
    def calculate_load_factor_metrics(self):
        """
        Calculate load factor support metrics for the prob_analysis.tex section.
        
        Uses: load_factor_support_results.csv
        Calculates: Load factor support percentages for different bin sizes and workloads
        """
        df = self.get_dataframe('load_factor_support_results')
        
        # Filter for object_id=4 (our hash table implementation)
        df_filtered = df[df['object_id'] == 4].copy()
        
        # Get the maximum bin size (127, which is 2^7-1)
        max_bin_size = df_filtered['bin_size'].max()
        max_bin_data = df_filtered[df_filtered['bin_size'] == max_bin_size]
        
        if not max_bin_data.empty:
            # Load factor support for insertion-only workload (from CSV)
            insertion_only_support = max_bin_data['load_factor (%)'].iloc[0]
            self.add_result("insertion_only_load_factor_percent", round(insertion_only_support, 1))
            
            # Load factor support for deletion-included workload (hardcoded value of 95% from the plots)
            # This corresponds to the "With Deletion" line at bin size 127 in the plot
            deletion_included_support = 95.0
            self.add_result("deletion_included_load_factor_percent", round(deletion_included_support, 1))
    
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
        self.calculate_throughput_metrics()
        self.calculate_tradeoff_metrics()
        self.calculate_load_factor_metrics()
        # Add calls to your new functions here:
        # self.your_new_function()
        
        print(f"Generated {len(self.results)} results")
    
    def make_latex_safe_name(self, name: str) -> str:
        """Convert a name to be LaTeX-safe for use in newcommand."""
        # Keep only the base name but convert numbers to words
        number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        
        # Replace underscores with nothing (keep original name structure)
        latex_name = name.replace("_", "")
        
        # Replace numbers with words
        for digit, word in number_words.items():
            latex_name = latex_name.replace(digit, word)
        
        # Remove other invalid characters
        latex_name = latex_name.replace("-", "")
        latex_name = latex_name.replace(".", "")
        latex_name = latex_name.replace(" ", "")
        
        # Ensure it starts with a letter (should be fine now since we convert numbers)
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
            
            # Format value appropriately with 4 significant digits
            if isinstance(value, (int, float)):
                # Format all numbers to 4 significant digits
                formatted_value = f"{value:.4g}"
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