import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import wofz
import os

# --- CONSTANTS ---
KB_EV = 8.617e-5         # Boltzmann constant (eV/K)
H_PLANCK = 4.135e-15     # Planck constant (eV s)
ME_KG = 9.109e-31        # Electron mass (kg)
EV_J = 1.602e-19         # eV to Joules

class LibsEngine:
    def __init__(self, db_path):
        if not os.path.exists(db_path):
            raise FileNotFoundError("Database not found.")
        self.conn = sqlite3.connect(db_path)
        
    def get_ionization_potential(self, element, sp_num):
        """Get IP from the species_physics table"""
        query = "SELECT ip_ev FROM species_physics WHERE element=? AND sp_num=?"
        cur = self.conn.cursor()
        cur.execute(query, (element, sp_num))
        res = cur.fetchone()
        return res[0] if res else None

    def get_lines(self, element, sp_num, min_wl, max_wl):
        query = """
            SELECT wavelength_nm, aki, ek_ev, gk 
            FROM lines 
            WHERE element = ? AND sp_num = ? 
            AND wavelength_nm BETWEEN ? AND ?
        """
        return pd.read_sql_query(query, self.conn, params=(element, sp_num, min_wl, max_wl))

    def saha_eggert_ratio(self, element, T_eV, Ne_cm3):
        """
        Calculates N_II / N_I ratio using Saha-Eggert Equation.
        
        N_II / N_I = (2 / Ne) * (2*pi*me*k*T / h^2)^1.5 * exp(-E_ion / kT)
        
        This tells us if the plasma is mostly Neutral (Ratio < 1) or Ionized (Ratio > 1).
        """
        # 1. Get Ionization Energy (Neutral -> Singly Ionized)
        # We assume sp_num=1 (Neutral) calculates ratio relative to sp_num=2 (Ion)
        ip_ev = self.get_ionization_potential(element, 1)
        if not ip_ev: return 0.0 # Missing data
        
        # 2. Convert Ne from cm^-3 to m^-3 (SI units required for quantum constants)
        Ne_m3 = Ne_cm3 * 1e6
        T_K = T_eV / KB_EV
        
        # 3. Constants for Saha (The "2.4e21" factor is derived from constants)
        # Simplified form: ratio = ( 6.04e21 / Ne_cm3 ) * T_eV^1.5 * exp(-IP/T_eV)
        # Note: Assuming Partition Function ratio Z_II/Z_I approx 1.0 (Standard approximation without full Z tables)
        
        saha_factor = (6.04e21 / Ne_cm3) * (T_eV**1.5) * np.exp(-ip_ev / T_eV)
        return saha_factor

    def generate_spectrum(self, element, T_eV, Ne_cm3, min_wl, max_wl):
        """
        Generates a dual-species spectrum (Neutral + Ion) weighted by Saha Balance.
        """
        # 1. Calculate Balance
        ion_neutral_ratio = self.saha_eggert_ratio(element, T_eV, Ne_cm3)
        
        # Normalize populations: N_I + N_II = 1.0
        frac_I = 1.0 / (1.0 + ion_neutral_ratio)
        frac_II = 1.0 - frac_I
        
        print(f"--- Plasma Condition: {element} @ {T_eV}eV, Ne={Ne_cm3:.1e} ---")
        print(f"Ionization Potential: {self.get_ionization_potential(element, 1)} eV")
        print(f"Saha Ratio (II/I):    {ion_neutral_ratio:.3f}")
        print(f"Composition:          {frac_I*100:.1f}% Neutral | {frac_II*100:.1f}% Ion")

        # 2. Get Lines
        df_I = self.get_lines(element, 1, min_wl, max_wl)
        df_II = self.get_lines(element, 2, min_wl, max_wl)
        
        # 3. Compute Intensities (Boltzmann + Saha Weighting)
        x_grid = np.linspace(min_wl, max_wl, 10000)
        y_grid = np.zeros_like(x_grid)
        
        # Process Neutrals
        if not df_I.empty:
            df_I['I'] = frac_I * (df_I['gk'] * df_I['aki'] / df_I['wavelength_nm']) * np.exp(-df_I['ek_ev'] / T_eV)
            y_grid += self._render_lines(df_I, x_grid)
            
        # Process Ions
        if not df_II.empty:
            df_II['I'] = frac_II * (df_II['gk'] * df_II['aki'] / df_II['wavelength_nm']) * np.exp(-df_II['ek_ev'] / T_eV)
            y_grid += self._render_lines(df_II, x_grid)
            
        return x_grid, y_grid

    def _render_lines(self, df, x_grid):
        """Render lines onto grid using Voigt profiles"""
        y = np.zeros_like(x_grid)
        # Skip lines too weak to matter (Optimization)
        max_I = df['I'].max()
        for _, line in df[df['I'] > max_I * 0.001].iterrows():
            # Physics: Instrument width + simplistic Stark
            sigma = 0.05 
            gamma = 0.02 
            
            # Vectorized Voigt
            z = (x_grid - line['wavelength_nm'] + 1j*gamma) / (sigma * np.sqrt(2))
            profile = line['I'] * np.real(wofz(z))
            y += profile
        return y

# --- MAIN ---
def run_simulation():
    engine = LibsEngine("libs_production.db")
    
    # User Inputs
    target = "Ti" # Try a Refractory or Transition metal!
    range_min, range_max = 300, 500
    
    # Compare Two Temperatures (e.g., Plasma Cooling)
    # 1. Hot Plasma (Early time)
    x, y_hot = engine.generate_spectrum(target, T_eV=1.5, Ne_cm3=1e17, min_wl=range_min, max_wl=range_max)
    
    # 2. Cool Plasma (Late time)
    _, y_cool = engine.generate_spectrum(target, T_eV=0.6, Ne_cm3=1e17, min_wl=range_min, max_wl=range_max)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Normalize for comparison
    if y_hot.max() > 0: y_hot /= y_hot.max()
    if y_cool.max() > 0: y_cool /= y_cool.max()

    plt.plot(x, y_hot + 0.1, label=f'{target} Hot (1.5 eV) - Mostly Ionic', color='tomato')
    plt.plot(x, y_cool, label=f'{target} Cool (0.6 eV) - Mostly Neutral', color='dodgerblue')
    
    plt.title(f"Saha-Boltzmann Evolution: {target} (I vs II)")
    plt.xlabel("Wavelength (nm)")
    plt.yticks([])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if os.path.exists("libs_production.db"):
        run_simulation()
    else:
        print("Please run the Database Builder first!")
