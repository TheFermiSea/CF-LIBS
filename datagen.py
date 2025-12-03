# --- COLAB SETUP ---
!pip install requests-cache ASDCache

import sqlite3
import pandas as pd
import requests
import requests_cache
import re
import time
import sys
import os

# --- IMPORT ASDCACHE ---
possible_paths = ['ASDCache/src', 'src', 'antoinetue/asdcache/ASDCache-7c5d709e6c655311993616700e8a23d7e4cfb1fb/src']
for path in possible_paths:
    if os.path.exists(path):
        sys.path.append(os.path.abspath(path))
        break

try:
    from ASDCache import SpectraCache
except ImportError:
    print("CRITICAL: ASDCache not found. Please upload the folder.")
    sys.exit(1)

# --- CONFIGURATION: PRODUCTION GRADE ---
# Full Periodic Table (Practical Subset for Spectroscopy)
# Excludes short-lived radioactives (Tc, Pm) and heavy actinides > U
# Ordered by Atomic Number (Z)
ALL_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
    "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Th", "U"
]

# For Ultrafast LIBS, early plasma is HOT. We need up to Stage IV.
STAGES = ["I", "II", "III", "IV"] 

DB_NAME = "libs_production.db"
CM_TO_EV = 1.23984193e-4 

# Separate cache for Ionization Energy (NIST IE Form)
ie_session = requests_cache.CachedSession('nist_ie_cache', expire_after=2592000) # 30 days

def fetch_ionization_potential(element, stage_roman):
    """
    Scrapes the NIST IE database. Critical for Saha-Eggert calculations.
    """
    url = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"
    params = {'spectra': f"{element} {stage_roman}", 'units': 1, 'format': 3, 'submit': 'Retrieve Data'}
    
    try:
        response = ie_session.get(url, params=params)
        # Parse text format: "Element Stage   Energy"
        for line in response.text.splitlines():
            if line.strip().startswith(element):
                parts = line.split()
                for part in parts:
                    clean = re.sub(r'[()\[\]]', '', part) # Remove uncertainties
                    try:
                        val = float(clean)
                        if val > 0 and val < 5000: # Sanity check
                            return val
                    except ValueError:
                        continue
        return None
    except:
        return None

def build_production_db():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    
    conn = sqlite3.connect(DB_NAME)
    
    # 1. SPECTRA TABLE (The Lines)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS lines (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            wavelength_nm REAL,
            aki REAL,
            ei_ev REAL,
            ek_ev REAL,
            gi REAL,
            gk REAL,
            rel_int REAL,
            UNIQUE(element, sp_num, wavelength_nm, ek_ev)
        )
    ''')

    # 2. PHYSICS TABLE (The Constants)
    # Stores Ionization Potentials (IP) needed for Saha Equation
    conn.execute('''
        CREATE TABLE IF NOT EXISTS species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
    ''')
    
    nist = SpectraCache()
    total_lines = 0
    
    print(f"--- STARTING PRODUCTION HARVEST: {len(ALL_ELEMENTS)} Elements ---")
    
    for el in ALL_ELEMENTS:
        print(f"\nProcessing {el}...", end=" ")
        
        for stage in STAGES:
            sp_int = {"I":1, "II":2, "III":3, "IV":4}[stage]
            query = f"{el} {stage}"
            
            # A. Get Ionization Potential (IP)
            ip = fetch_ionization_potential(el, stage)
            if ip:
                conn.execute("INSERT OR REPLACE INTO species_physics VALUES (?,?,?)", (el, sp_int, ip))
                print(f"[{stage} IP:{ip:.1f}eV]", end=" ")
            
            # B. Get Spectra
            try:
                # Wide range for all spectrometers (UV to NIR)
                df = nist.fetch(query, wl_range=(150, 1100))
                
                if df.empty: continue

                # Strict Physics Filter: We ONLY want lines usable for calculations
                mask = df['obs_wl_air(nm)'].notna() & df['Aki(s^-1)'].notna() & df['Ek(cm-1)'].notna()
                clean = df[mask].copy()
                
                if clean.empty: continue
                
                # Convert Units
                sql_df = pd.DataFrame({
                    'element': el,
                    'sp_num': sp_int,
                    'wavelength_nm': clean['obs_wl_air(nm)'],
                    'aki': clean['Aki(s^-1)'],
                    'ei_ev': clean['Ei(cm-1)'] * CM_TO_EV,
                    'ek_ev': clean['Ek(cm-1)'] * CM_TO_EV,
                    'gi': clean['g_i'],
                    'gk': clean['g_k'],
                    'rel_int': pd.to_numeric(clean['intens'], errors='coerce').fillna(0)
                })
                
                # Deduplicate (NIST often lists 'Observed' and 'Ritz' as separate rows)
                sql_df = sql_df.drop_duplicates(subset=['wavelength_nm', 'ek_ev'])
                
                sql_df.to_sql('lines', conn, if_exists='append', index=False)
                total_lines += len(sql_df)
                print(f".", end="") # Dot indicates success for this stage
                
            except Exception as e:
                pass # Silent skip on errors to keep production running
                
        conn.commit()

    # Optimize
    conn.execute("CREATE INDEX IF NOT EXISTS idx_main ON lines(element, sp_num, wavelength_nm)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_phys ON species_physics(element, sp_num)")
    conn.close()
    
    print(f"\n\n--- HARVEST COMPLETE ---")
    print(f"Database: {DB_NAME}")
    print(f"Total Spectral Lines: {total_lines}")

if __name__ == "__main__":
    build_production_db()
