# cf_libs_analyzer.py
import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

# --- PHYSICS CONSTANTS ---
KB = 8.617e-5  # Boltzmann eV/K

class CFLIBS_Analyzer:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.elements = [] # List of detected elements
        
    def load_spectrum(self, wavelengths, intensities):
        """
        Load experimental data (1D arrays).
        Preprocessing: Baseline removal and Peak Finding.
        """
        self.wl = np.array(wavelengths)
        self.intensity = np.array(intensities) - self._estimate_baseline(intensities)
        self.peaks = self._find_peaks(self.wl, self.intensity)
        print(f" Spectrum Loaded: {len(self.peaks)} peaks found.")
        
    def _estimate_baseline(self, y, window=100):
        """Simple rolling minimum baseline subtraction"""
        return pd.Series(y).rolling(window, center=True).min().fillna(0).values

    def _find_peaks(self, x, y, threshold=0.05):
        """
        Basic peak finding. In production, use scipy.signal.find_peaks
        Returns DataFrame: [wavelength, intensity]
        """
        from scipy.signal import find_peaks
        height = y.max() * threshold
        indices, _ = find_peaks(y, height=height, distance=5)
        return pd.DataFrame({
            'wavelength': x[indices],
            'intensity': y[indices]
        })

    def identify_elements(self, search_list=None, tolerance_nm=0.1):
        """
        Matches experimental peaks to DB.
        If search_list is None, searches all DB (Slow!). 
        Better to provide expected alloys (e.g. ['Fe', 'Ni', 'Cr'])
        """
        if not search_list:
            search_list = ["Fe", "Al", "Ti", "Mn", "Si", "Mg", "Cu", "Ni", "Cr"] # Common LPBF

        found_elements = set()
        self.identified_lines = []

        print("Identifying Elements...")
        for el in search_list:
            # Query strong lines for this element
            query = """
                SELECT * FROM lines 
                WHERE element = ? AND sp_num = 1 
                AND rel_int > 100 
                ORDER BY rel_int DESC LIMIT 50
            """
            db_lines = pd.read_sql_query(query, self.conn, params=(el,))
            
            # Match
            hits = 0
            for _, db_line in db_lines.iterrows():
                # Check if we have a peak near this line
                match = self.peaks[
                    (self.peaks['wavelength'] > db_line['wavelength_nm'] - tolerance_nm) & 
                    (self.peaks['wavelength'] < db_line['wavelength_nm'] + tolerance_nm)
                ]
                if not match.empty:
                    hits += 1
                    # Store match for Solver
                    row = db_line.to_dict()
                    row['experimental_intensity'] = match.iloc[0]['intensity']
                    self.identified_lines.append(row)

            if hits > 3: # Threshold to confirm element presence
                found_elements.add(el)
        
        self.elements = list(found_elements)
        self.line_data = pd.DataFrame(self.identified_lines)
        print(f"Detected: {self.elements}")

    def calculate_partition_function(self, element, sp_num, T_eV):
        """
        Calculates Z(T) = sum(g * exp(-E/kT)) using scraped levels.
        """
        query = "SELECT g_level, energy_ev FROM energy_levels WHERE element=? AND sp_num=?"
        levels = pd.read_sql_query(query, self.conn, params=(element, sp_num))
        
        if levels.empty: return 1.0 # Fallback
        
        # Z = sum(g_i * exp(-E_i / (k*T)))
        Z = np.sum(levels['g_level'] * np.exp(-levels['energy_ev'] / T_eV))
        return Z

    def solve_cf_libs(self, initial_T=1.0):
        """
        THE CORE ALGORITHM:
        Iteratively solves for C_s (Concentration) and T_eV.
        """
        if self.line_data.empty:
            print("No lines identified. Cannot solve.")
            return

        print(f"Starting CF-LIBS Iteration (Initial T={initial_T} eV)...")
        
        # 1. Prepare Data
        df = self.line_data.copy()
        
        # Loop Variables
        T_eV = initial_T
        concentrations = {el: 1.0/len(self.elements) for el in self.elements}
        
        for iteration in range(10): # Iterative refinement
            
            # --- A. Update Partition Functions ---
            Z_map = {}
            for el in self.elements:
                Z_map[f"{el}_1"] = self.calculate_partition_function(el, 1, T_eV)
                # Ideally add Stage II here too using Saha-Eggert
            
            # --- B. Boltzmann Plot (Linearization) ---
            # ln(I_exp / (gA/Z)) = -E_upper / kT + ln(F * Concentration)
            # We normalize everything to calculate T first
            
            # We use Fe or Ti (rich spectra) to fix Temperature
            ref_el = "Fe" if "Fe" in self.elements else self.elements[0]
            ref_data = df[df['element'] == ref_el]
            
            if len(ref_data) > 5:
                # X = E_upper
                # Y = ln(I_exp * Z / (g * A))
                # Slope = -1/kT
                
                # Note: Wavelength factor included in Boltzmann?
                # Intensity I = F * C * (A * g / Z) * exp(-E/kT) * (1/lambda?)
                # Depends on if experimental I is energy or photons. Assuming Energy (standard).
                
                X = ref_data['ek_ev'].values
                Y = np.log(
                    (ref_data['experimental_intensity'] * Z_map[f"{ref_el}_1"]) / 
                    (ref_data['gk'] * ref_data['aki']) 
                )
                
                # Remove outliers (simple RANSAC or sigma clip)
                # ... [Insert Outlier Code Here] ...
                
                slope, intercept = np.polyfit(X, Y, 1)
                
                new_T = -1.0 / slope
                
                # Damping to prevent oscillation
                T_eV = 0.7 * T_eV + 0.3 * new_T
                
                # --- C. Calculate Concentrations ---
                # Intercept = ln(F * C_s)  -> but F is unknown.
                # We calculate Relative Factor Q_s = exp(intercept) = F * C_s
                # Then sum(C_s) = 1 to find F.
                
                qs_values = {}
                for el in self.elements:
                    el_data = df[df['element'] == el]
                    if el_data.empty: continue
                    
                    # Compute average F*C for this element
                    # F*C = I_exp * Z / (g * A * exp(-E/kT))
                    val = (el_data['experimental_intensity'] * Z_map[f"{el}_1"]) / \
                          (el_data['gk'] * el_data['aki'] * np.exp(-el_data['ek_ev'] / T_eV))
                    qs_values[el] = np.median(val) # Median is robust to outliers
                
                # Closure: sum(C_s) = 1
                # Q_total = F * sum(C_s) = F * 1 = F
                F_factor = sum(qs_values.values())
                
                for el in qs_values:
                    concentrations[el] = qs_values[el] / F_factor
                
                print(f" Iter {iteration}: T={T_eV:.3f} eV | Fe={concentrations.get('Fe',0):.1%}")
                
            else:
                print("Not enough lines for Reference Element to calculate T.")
                break

        print("\n--- FINAL RESULTS ---")
        print(f"Plasma Temperature: {T_eV:.3f} eV ({T_eV * 11604:.0f} K)")
        print("Composition:")
        for el, conc in concentrations.items():
            print(f"  {el}: {conc*100:.2f} %")

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    analyzer = CFLIBS_Analyzer("libs_production.db")
    
    # 1. Simulate a spectrum (Using your saha-eggert script or loading a CSV)
    # Here we create a dummy "experimental" input for demonstration
    # (In real life, load_spectrum from your CSV file)
    sim_wl = np.linspace(200, 500, 3000)
    sim_intensity = np.random.normal(0, 1, 3000) # Noise
    analyzer.load_spectrum(sim_wl, sim_intensity)
    
    # 2. Run Analysis
    analyzer.identify_elements(search_list=["Fe", "Cr", "Ni"])
    analyzer.solve_cf_libs()
