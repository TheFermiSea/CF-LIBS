# datagen_v2.py
# --- COLAB SETUP ---
# !pip install requests-cache ASDCache

import sqlite3
import pandas as pd
import requests_cache
import re
import sys
import os
import io

# --- CONFIGURATION ---
DB_NAME = "libs_production.db"
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
STAGES = ["I", "II", "III"] # IV is rarely needed for Z(T) in standard LIBS

# Cache for Levels (Long expiry as NIST levels rarely change)
levels_session = requests_cache.CachedSession('nist_levels_cache', expire_after=None)

def fetch_energy_levels(element, stage_roman):
    """
    Scrapes the NIST Atomic Levels form. 
    Essential for calculating Partition Functions Z(T).
    """
    url = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"
    query = f"{element} {stage_roman}"
    params = {
        'spectrum': query,
        'units': 1, 'format': 3, 'multiplet_ordered': 0,
        'conf_out': 'off', 'term_out': 'off', 'level_out': 'on',
        'unc_out': 0, 'j_out': 'on', 'g_out': 'on', 'land_out': 'off',
        'submit': 'Retrieve Data'
    }
    
    try:
        response = levels_session.get(url, params=params)
        # NIST returns TSV-like text. We parse it robustly.
        lines = response.text.splitlines()
        data = []
        
        # Regex to capture: Configuration, Term, J, g, Level(eV)
        # We only really care about g (statistical weight) and Level (energy)
        for line in lines:
            if "Level" in line and "eV" in line: continue # Skip header
            
            # Simple parsing strategy: look for the energy column (usually last)
            # and g column (usually 3rd or 4th)
            # This is fragile, so we use a robust pandas read if possible, 
            # but NIST's output is messy. Here is a direct parse:
            
            # Split by tabs or multiple spaces
            parts = re.split(r'\s+', line.strip())
            if len(parts) < 3: continue
            
            try:
                # Find Energy (Level) - often the last number
                # Strip brackets/parentheses for theoretical levels
                clean_parts = [re.sub(r'[\[\]\(\)\?]', '', p) for p in parts]
                
                # Iterate backwards to find Energy
                energy = None
                for p in reversed(clean_parts):
                    try: 
                        energy = float(p)
                        break
                    except: continue
                
                if energy is None: continue
                
                # Find g (Statistical Weight) - usually integer, often near the end
                # but before energy.
                g = None
                # Heuristic: Scan remaining parts for integer g
                for p in clean_parts:
                    if p.isdigit():
                        val = int(p)
                        # g is usually small (<100)
                        if val > 0 and val < 200: 
                            g = val
                            # Don't break immediately, might be J. 
                            # But NIST usually puts J then g. 
                            # We take the last valid integer before Energy as g.
                
                if g and energy >= 0:
                    data.append((g, energy))
                    
            except: continue
            
        return data # List of (g, energy_ev)
        
    except Exception as e:
        print(f" [Levels Error: {e}]", end="")
        return []

def build_production_db():
    # ... [Keep your existing Line Fetching code here] ...
    # This function appends the Level fetching logic
    
    conn = sqlite3.connect(DB_NAME)
    
    # NEW TABLE: Energy Levels
    conn.execute('''
        CREATE TABLE IF NOT EXISTS energy_levels (
            element TEXT,
            sp_num INTEGER,
            g_level INTEGER,
            energy_ev REAL
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_levels ON energy_levels(element, sp_num)')
    
    print("--- UPDATING DATABASE WITH PARTITION FUNCTION DATA ---")
    
    for el in ALL_ELEMENTS:
        print(f"\nProcessing Levels for {el}...", end=" ")
        for stage in STAGES:
            sp_int = {"I":1, "II":2, "III":3, "IV":4}.get(stage, 0)
            if sp_int == 0: continue
            
            levels = fetch_energy_levels(el, stage)
            if levels:
                # Batch Insert
                rows = [(el, sp_int, g, en) for (g, en) in levels]
                conn.executemany("INSERT INTO energy_levels VALUES (?,?,?,?)", rows)
                print(f"[{stage}: {len(rows)} levels]", end=" ")
            else:
                print(f"[{stage}: 0]", end=" ")
        conn.commit()
    
    conn.close()
    print("\n\nDatabase Update Complete.")

if __name__ == "__main__":
    # Run this AFTER your existing datagen.py to add the levels
    build_production_db()
