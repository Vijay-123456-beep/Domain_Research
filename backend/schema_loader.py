"""
Schema Loader Module
Loads and manages domain-specific schemas with units and validation rules.
"""
import json
import os
import re
from typing import Dict, List, Optional, Any, Tuple


class SchemaLoader:
    """
    Loads domain schemas that define:
    - Attributes with expected units, ranges, and aliases
    - Unit ontology with conversions and dimensions
    - Validation rules for each attribute
    """
    
    def __init__(self, schema_path: Optional[str] = None, workspace: Optional[str] = None):
        self.schema: Dict[str, Any] = {}
        self.unit_ontology: Dict[str, Any] = {}
        self.attribute_index: Dict[str, str] = {}  # alias -> canonical name
        self.workspace = workspace
        
        # Initialize with comprehensive unit ontology
        self._init_generic_unit_ontology()
        
        if schema_path and os.path.exists(schema_path):
            self.load_schema(schema_path)
        elif workspace:
            # Try to load or generate schema from workspace
            self.load_or_create_domain_schema(workspace)
    
    def _init_generic_unit_ontology(self):
        """
        Initialize generic unit ontology that works across ALL scientific domains.
        Covers: Energy, Magnetics, Solar, Batteries, Biomedical, Chemistry, Physics, etc.
        """
        self.unit_ontology = {
            'dimensions': {
                # Energy & Power (batteries, supercapacitors, solar)
                'energy': {
                    'base_unit': 'J',
                    'aliases': ['J', 'kJ', 'MJ', 'Wh', 'kWh', 'MWh', 'mWh', 'cal', 'kcal', 'eV', 'keV', 'MeV'],
                    'conversion': {'kJ': 1000, 'MJ': 1e6, 'Wh': 3600, 'kWh': 3.6e6, 'mWh': 3.6, 'cal': 4.184, 'kcal': 4184, 'eV': 1.602e-19}
                },
                'energy_per_mass': {
                    'base_unit': 'J/kg',
                    'aliases': ['J/kg', 'kJ/kg', 'MJ/kg', 'Wh/kg', 'kWh/kg', 'mWh/g', 'mWh/kg', 'cal/g'],
                    'conversion': {'kJ/kg': 1000, 'MJ/kg': 1e6, 'Wh/kg': 3600, 'kWh/kg': 3.6e6, 'mWh/g': 3.6, 'cal/g': 4184}
                },
                'energy_per_volume': {
                    'base_unit': 'J/m³',
                    'aliases': ['J/m³', 'J/L', 'kJ/L', 'Wh/L', 'kWh/L', 'mWh/cm³', 'mWh/mL'],
                    'conversion': {'J/L': 1000, 'kJ/L': 1e6, 'Wh/L': 3600, 'kWh/L': 3.6e6, 'mWh/cm³': 3600}
                },
                'power': {
                    'base_unit': 'W',
                    'aliases': ['W', 'kW', 'MW', 'mW', 'μW', 'uW', 'nW', 'TW', 'GW', 'horsepower', 'hp'],
                    'conversion': {'kW': 1000, 'MW': 1e6, 'mW': 0.001, 'μW': 1e-6, 'uW': 1e-6, 'nW': 1e-9, 'TW': 1e12, 'GW': 1e9, 'horsepower': 745.7, 'hp': 745.7}
                },
                'power_per_mass': {
                    'base_unit': 'W/kg',
                    'aliases': ['W/kg', 'kW/kg', 'MW/kg', 'W/g', 'mW/g', 'kW/g'],
                    'conversion': {'kW/kg': 1000, 'MW/kg': 1e6, 'W/g': 1000, 'mW/g': 1, 'kW/g': 1e6}
                },
                
                # Electrical (batteries, solar cells, electronics)
                'voltage': {
                    'base_unit': 'V',
                    'aliases': ['V', 'mV', 'kV', 'MV', 'μV', 'uV', 'nV', 'pV', 'volts', 'millivolts'],
                    'conversion': {'mV': 0.001, 'kV': 1000, 'MV': 1e6, 'μV': 1e-6, 'uV': 1e-6, 'nV': 1e-9, 'pV': 1e-12}
                },
                'current': {
                    'base_unit': 'A',
                    'aliases': ['A', 'mA', 'kA', 'MA', 'μA', 'uA', 'nA', 'pA', 'fA', 'amperes', 'amps'],
                    'conversion': {'mA': 0.001, 'kA': 1000, 'MA': 1e6, 'μA': 1e-6, 'uA': 1e-6, 'nA': 1e-9, 'pA': 1e-12, 'fA': 1e-15}
                },
                'current_per_mass': {
                    'base_unit': 'A/kg',
                    'aliases': ['A/kg', 'A/g', 'mA/g', 'mA/kg', 'kA/kg', 'mA/mg'],
                    'conversion': {'A/g': 1000, 'mA/g': 1, 'mA/kg': 0.001, 'kA/kg': 1000, 'mA/mg': 1}
                },
                'current_per_area': {
                    'base_unit': 'A/m²',
                    'aliases': ['A/m²', 'A/cm²', 'mA/cm²', 'μA/cm²', 'uA/cm²', 'A/mm²', 'mA/mm²'],
                    'conversion': {'A/cm²': 1e4, 'mA/cm²': 10, 'μA/cm²': 0.01, 'uA/cm²': 0.01, 'A/mm²': 1e6, 'mA/mm²': 1000}
                },
                'resistance': {
                    'base_unit': 'Ω',
                    'aliases': ['Ω', 'ohm', 'ohms', 'mΩ', 'kΩ', 'MΩ', 'GΩ', 'TΩ'],
                    'conversion': {'mΩ': 0.001, 'kΩ': 1000, 'MΩ': 1e6, 'GΩ': 1e9, 'TΩ': 1e12, 'ohm': 1, 'ohms': 1}
                },
                'capacitance': {
                    'base_unit': 'F',
                    'aliases': ['F', 'mF', 'μF', 'uF', 'nF', 'pF', 'fF', 'farad', 'farads'],
                    'conversion': {'mF': 0.001, 'μF': 1e-6, 'uF': 1e-6, 'nF': 1e-9, 'pF': 1e-12, 'fF': 1e-15}
                },
                'capacitance_per_mass': {
                    'base_unit': 'F/kg',
                    'aliases': ['F/kg', 'F/g', 'mF/g', 'μF/g', 'uF/g', 'mF/kg'],
                    'conversion': {'F/g': 1000, 'mF/g': 1, 'μF/g': 0.001, 'uF/g': 0.001, 'mF/kg': 0.001}
                },
                'conductivity': {
                    'base_unit': 'S/m',
                    'aliases': ['S/m', 'S/cm', 'mS/cm', 'μS/cm', 'uS/cm', 'mS/m'],
                    'conversion': {'S/cm': 100, 'mS/cm': 0.1, 'μS/cm': 1e-4, 'uS/cm': 1e-4, 'mS/m': 0.001}
                },
                'electric_field': {
                    'base_unit': 'V/m',
                    'aliases': ['V/m', 'kV/m', 'MV/m', 'V/cm', 'V/mm'],
                    'conversion': {'kV/m': 1000, 'MV/m': 1e6, 'V/cm': 100, 'V/mm': 1000}
                },
                'charge': {
                    'base_unit': 'C',
                    'aliases': ['C', 'mC', 'μC', 'uC', 'nC', 'pC', 'coulomb', 'coulombs', 'Ah', 'mAh'],
                    'conversion': {'mC': 0.001, 'μC': 1e-6, 'uC': 1e-6, 'nC': 1e-9, 'pC': 1e-12, 'Ah': 3600, 'mAh': 3.6}
                },
                
                # Magnetic (magnetics, SPIONs, MRI)
                'magnetic_field': {
                    'base_unit': 'T',
                    'aliases': ['T', 'mT', 'μT', 'uT', 'nT', 'kT', 'MT', 'tesla', 'gauss', 'G'],
                    'conversion': {'mT': 0.001, 'μT': 1e-6, 'uT': 1e-6, 'nT': 1e-9, 'kT': 1000, 'MT': 1e6, 'gauss': 1e-4, 'G': 1e-4}
                },
                'magnetic_field_strength': {
                    'base_unit': 'A/m',
                    'aliases': ['A/m', 'kA/m', 'MA/m', 'Oe', 'oersted'],
                    'conversion': {'kA/m': 1000, 'MA/m': 1e6, 'Oe': 79.577, 'oersted': 79.577}
                },
                'magnetization': {
                    'base_unit': 'A/m',
                    'aliases': ['A/m', 'kA/m', 'MA/m', 'emu/cm³', 'emu/g', 'J/T·m³'],
                    'conversion': {'kA/m': 1000, 'MA/m': 1e6, 'emu/cm³': 1000, 'emu/g': 1}
                },
                'specific_absorption_rate': {
                    'base_unit': 'W/kg',
                    'aliases': ['W/kg', 'mW/g', 'W/g', 'kW/kg'],
                    'conversion': {'mW/g': 1, 'W/g': 1000, 'kW/kg': 1000}
                },
                'frequency': {
                    'base_unit': 'Hz',
                    'aliases': ['Hz', 'kHz', 'MHz', 'GHz', 'THz', 'mHz', 'Hz'],
                    'conversion': {'kHz': 1000, 'MHz': 1e6, 'GHz': 1e9, 'THz': 1e12, 'mHz': 0.001}
                },
                
                # Optical (solar cells, photonics)
                'wavelength': {
                    'base_unit': 'm',
                    'aliases': ['m', 'nm', 'μm', 'um', 'mm', 'cm', 'km', 'Å', 'angstrom', 'pm'],
                    'conversion': {'nm': 1e-9, 'μm': 1e-6, 'um': 1e-6, 'mm': 0.001, 'cm': 0.01, 'km': 1000, 'Å': 1e-10, 'angstrom': 1e-10, 'pm': 1e-12}
                },
                'irradiance': {
                    'base_unit': 'W/m²',
                    'aliases': ['W/m²', 'W/cm²', 'mW/cm²', 'kW/m²', 'sun', 'suns'],
                    'conversion': {'W/cm²': 1e4, 'mW/cm²': 10, 'kW/m²': 1000, 'sun': 1000, 'suns': 1000}
                },
                'efficiency': {
                    'base_unit': '%',
                    'aliases': ['%', 'percent', 'fraction'],
                    'conversion': {'fraction': 100, 'percent': 1}
                },
                'quantum_yield': {
                    'base_unit': '%',
                    'aliases': ['%', 'percent', 'fraction'],
                    'conversion': {'fraction': 100}
                },
                'absorbance': {
                    'base_unit': 'AU',
                    'aliases': ['AU', 'OD', 'optical density', 'a.u.', 'absorbance units'],
                    'conversion': {}
                },
                
                # Geometric (all domains)
                'length': {
                    'base_unit': 'm',
                    'aliases': ['m', 'km', 'cm', 'mm', 'μm', 'um', 'nm', 'pm', 'fm', 'Å', 'angstrom', 'inch', 'ft', 'yd', 'mi'],
                    'conversion': {'km': 1000, 'cm': 0.01, 'mm': 0.001, 'μm': 1e-6, 'um': 1e-6, 'nm': 1e-9, 'pm': 1e-12, 'fm': 1e-15, 'Å': 1e-10, 'angstrom': 1e-10, 'inch': 0.0254, 'ft': 0.3048, 'yd': 0.9144, 'mi': 1609.34}
                },
                'area': {
                    'base_unit': 'm²',
                    'aliases': ['m²', 'm2', 'km²', 'cm²', 'cm2', 'mm²', 'nm²', 'ft²', 'in²', 'acre', 'hectare', 'ha'],
                    'conversion': {'km²': 1e6, 'cm²': 1e-4, 'cm2': 1e-4, 'mm²': 1e-6, 'nm²': 1e-18, 'ft²': 0.0929, 'in²': 0.000645, 'acre': 4046.86, 'hectare': 1e4, 'ha': 1e4}
                },
                'volume': {
                    'base_unit': 'm³',
                    'aliases': ['m³', 'm3', 'L', 'mL', 'kL', 'cm³', 'cm3', 'cc', 'mm³', 'ft³', 'gal', 'liter', 'liters', 'litre'],
                    'conversion': {'L': 0.001, 'mL': 1e-6, 'kL': 1, 'cm³': 1e-6, 'cm3': 1e-6, 'cc': 1e-6, 'mm³': 1e-9, 'ft³': 0.0283, 'gal': 0.003785, 'liter': 0.001, 'liters': 0.001, 'litre': 0.001}
                },
                'area_per_mass': {
                    'base_unit': 'm²/kg',
                    'aliases': ['m²/kg', 'm²/g', 'cm²/g', 'm2/kg', 'm2/g'],
                    'conversion': {'m²/g': 1000, 'cm²/g': 0.0001, 'm2/g': 1000, 'm2/kg': 1}
                },
                'volume_per_mass': {
                    'base_unit': 'm³/kg',
                    'aliases': ['m³/kg', 'm³/g', 'cm³/g', 'cc/g', 'mL/g', 'L/kg'],
                    'conversion': {'m³/g': 1000, 'cm³/g': 1e-6, 'cc/g': 1e-6, 'mL/g': 1e-6, 'L/kg': 0.001}
                },
                
                # Mass & Concentration (chemistry, biology, materials)
                'mass': {
                    'base_unit': 'kg',
                    'aliases': ['kg', 'g', 'mg', 'μg', 'ug', 'ng', 'pg', 'fg', 'ton', 'tonne', 'lb', 'oz', 'grain'],
                    'conversion': {'g': 0.001, 'mg': 1e-6, 'μg': 1e-9, 'ug': 1e-9, 'ng': 1e-12, 'pg': 1e-15, 'fg': 1e-18, 'ton': 1000, 'tonne': 1000, 'lb': 0.4536, 'oz': 0.02835}
                },
                'concentration': {
                    'base_unit': 'mol/m³',
                    'aliases': ['M', 'mM', 'μM', 'uM', 'nM', 'pM', 'fM', 'mol/L', 'mmol/L', 'mol/m³'],
                    'conversion': {'mM': 1, 'μM': 1e-3, 'uM': 1e-3, 'nM': 1e-6, 'pM': 1e-9, 'fM': 1e-12, 'mol/L': 1000, 'mmol/L': 1, 'mol/m³': 1}
                },
                'density': {
                    'base_unit': 'kg/m³',
                    'aliases': ['kg/m³', 'kg/m3', 'g/cm³', 'g/cm3', 'g/mL', 'kg/L'],
                    'conversion': {'kg/m3': 1, 'g/cm³': 1000, 'g/cm3': 1000, 'g/mL': 1000, 'kg/L': 1000}
                },
                
                # Thermal (all domains)
                'temperature': {
                    'base_unit': 'K',
                    'aliases': ['K', '°C', 'C', '℃', '°F', 'F', '℉'],
                    'conversion': {'°C': 1, 'C': 1, '℃': 1, '°F': 0.5556, 'F': 0.5556, '℉': 0.5556}
                },
                'thermal_conductivity': {
                    'base_unit': 'W/(m·K)',
                    'aliases': ['W/(m·K)', 'W/mK', 'W/(m·K)', 'W/cmK'],
                    'conversion': {'W/mK': 1, 'W/cmK': 100}
                },
                'specific_heat': {
                    'base_unit': 'J/(kg·K)',
                    'aliases': ['J/(kg·K)', 'kJ/(kg·K)', 'J/gK', 'cal/(g·°C)'],
                    'conversion': {'kJ/(kg·K)': 1000, 'J/gK': 1000, 'cal/(g·°C)': 4184}
                },
                
                # Time (universal)
                'time': {
                    'base_unit': 's',
                    'aliases': ['s', 'sec', 'second', 'seconds', 'ms', 'μs', 'us', 'ns', 'ps', 'fs', 'min', 'hr', 'h', 'day', 'week', 'month', 'year', 'yr'],
                    'conversion': {'ms': 0.001, 'μs': 1e-6, 'us': 1e-6, 'ns': 1e-9, 'ps': 1e-12, 'fs': 1e-15, 'min': 60, 'hr': 3600, 'h': 3600, 'day': 86400, 'week': 604800, 'month': 2.628e6, 'year': 3.154e7, 'yr': 3.154e7, 'sec': 1, 'second': 1, 'seconds': 1}
                },
                'rate': {
                    'base_unit': 'Hz',
                    'aliases': ['Hz', 's⁻¹', 's-1', '/s', 'per second', '1/s'],
                    'conversion': {'s⁻¹': 1, 's-1': 1, '/s': 1, 'per second': 1, '1/s': 1}
                },
                
                # Dimensionless
                'dimensionless': {
                    'base_unit': '',
                    'aliases': ['', '-', 'ratio', 'fraction', 'index', 'number'],
                    'conversion': {}
                },
                'percentage': {
                    'base_unit': '%',
                    'aliases': ['%', 'percent', 'wt%', 'at%', 'vol%', 'mol%'],
                    'conversion': {'percent': 1, 'wt%': 1, 'at%': 1, 'vol%': 1, 'mol%': 1}
                },
                'angle': {
                    'base_unit': 'rad',
                    'aliases': ['rad', 'radian', 'radians', '°', 'deg', 'degree', 'degrees', 'arcmin', 'arcsec'],
                    'conversion': {'°': 0.01745, 'deg': 0.01745, 'degree': 0.01745, 'degrees': 0.01745, 'arcmin': 0.00029, 'arcsec': 4.848e-6}
                },
                
                # Biomedical specific
                'cell_count': {
                    'base_unit': 'cells/mL',
                    'aliases': ['cells/mL', 'cells/μL', 'cells/uL', 'cells/mm³', 'CFU/mL', 'cells'],
                    'conversion': {'cells/μL': 1000, 'cells/uL': 1000, 'cells/mm³': 1000}
                },
                'dose': {
                    'base_unit': 'mg/kg',
                    'aliases': ['mg/kg', 'g/kg', 'μg/kg', 'ug/kg', 'mg/m²', 'mg', 'g', 'μg'],
                    'conversion': {'g/kg': 1000, 'μg/kg': 0.001, 'ug/kg': 0.001, 'mg/m²': 1, 'g': 1000, 'μg': 0.001, 'ug': 0.001}
                },
                'activity': {
                    'base_unit': 'Bq',
                    'aliases': ['Bq', 'kBq', 'MBq', 'GBq', 'Ci', 'mCi', 'μCi', 'uCi', 'becquerel'],
                    'conversion': {'kBq': 1000, 'MBq': 1e6, 'GBq': 1e9, 'Ci': 3.7e10, 'mCi': 3.7e7, 'μCi': 3.7e4, 'uCi': 3.7e4, 'becquerel': 1}
                },
                'pressure': {
                    'base_unit': 'Pa',
                    'aliases': ['Pa', 'kPa', 'MPa', 'GPa', 'bar', 'mbar', 'atm', 'torr', 'mmHg', 'psi'],
                    'conversion': {'kPa': 1000, 'MPa': 1e6, 'GPa': 1e9, 'bar': 1e5, 'mbar': 100, 'atm': 101325, 'torr': 133.322, 'mmHg': 133.322, 'psi': 6894.76}
                }
            }
        }

    def load_or_create_domain_schema(self, workspace: str):
        """
        Load schema from aliases.json or create dynamically for ANY domain.
        Supports: solar cells, magnetics, batteries, autism, biomedical, chemistry, etc.
        """
        alias_file = os.path.join(workspace, "aliases.json")
        
        if os.path.exists(alias_file):
            try:
                with open(alias_file, 'r', encoding='utf-8') as f:
                    aliases_data = json.load(f)
                
                # Generate schema dynamically from aliases
                self.schema = self._generate_schema_from_aliases(aliases_data)
                self._build_attribute_index()
                print(f"[SCHEMA] Generated dynamic schema from aliases.json with {len(self.schema)} attributes")
                return
                
            except Exception as e:
                print(f"[SCHEMA] Error loading aliases.json: {e}, using generic schema")
        
        # If no aliases file, create minimal generic schema
        self.schema = self._create_generic_schema()
        self._build_attribute_index()
        print(f"[SCHEMA] Using generic schema with {len(self.schema)} common scientific attributes")

    def _generate_schema_from_aliases(self, aliases_data: Dict) -> Dict[str, Any]:
        """
        Generate full schema from aliases.json format.
        Infer dimensions, ranges, and validation rules from units.
        """
        schema = {}
        
        for attr_name, attr_info in aliases_data.items():
            units = attr_info.get('units', [])
            aliases = attr_info.get('aliases', [])
            
            # Infer dimension from units
            dimension = self._infer_dimension_from_units(units)
            
            # Infer value type
            value_type = 'float'
            validation_rules = []
            
            # Check if categorical (no units or specific keywords)
            if not units or any(keyword in attr_name.lower() for keyword in 
                              ['type', 'method', 'material', 'name', 'species', 'category', 'grade']):
                value_type = 'string'
                validation_rules.append('not_numeric')
            else:
                validation_rules.append('unit_required')
            
            # Infer expected range from dimension
            expected_range = self._infer_range_from_dimension(dimension)
            
            # Check for ratio/percentage patterns
            if any('%' in str(u) for u in units) or 'ratio' in attr_name.lower():
                validation_rules.append('range_0_100')
                if not expected_range:
                    expected_range = {'min': 0, 'max': 100}
            
            schema[attr_name] = {
                'units': units,
                'aliases': aliases,
                'type': value_type,
                'dimension': dimension,
                'expected_range': expected_range,
                'validation_rules': validation_rules
            }
        
        return schema

    def _infer_dimension_from_units(self, units: List[str]) -> Optional[str]:
        """
        Infer physical dimension from list of units.
        Works across all scientific domains.
        """
        if not units:
            return None
        
        # Check each unit against our ontology
        for unit in units:
            unit_lower = unit.lower().replace('−', '-').replace('⁻', '-')
            
            # Direct dimension lookup
            for dim_name, dim_data in self.unit_ontology.get('dimensions', {}).items():
                # Check base unit
                if unit_lower == dim_data['base_unit'].lower():
                    return dim_name
                
                # Check aliases
                for alias in dim_data.get('aliases', []):
                    if unit_lower == alias.lower():
                        return dim_name
                    # Handle superscript variants
                    alias_norm = alias.lower().replace('−', '-').replace('⁻', '-')
                    if unit_lower == alias_norm:
                        return dim_name
        
        # Try pattern matching for compound units
        unit_str = ' '.join(units).lower()
        
        # Per-mass patterns (A/g, W/kg, etc.)
        if re.search(r'/\s*(g|kg|mg|μg|ug)\b', unit_str):
            if 'v' in unit_str or 'volt' in unit_str:
                return 'voltage'  # or voltage_per_mass
            if 'a' in unit_str or 'amp' in unit_str:
                return 'current_per_mass'
            if 'w' in unit_str or 'watt' in unit_str:
                return 'power_per_mass'
            if 'f' in unit_str or 'farad' in unit_str:
                return 'capacitance_per_mass'
            if 'j' in unit_str or 'wh' in unit_str:
                return 'energy_per_mass'
        
        # Per-area patterns
        if re.search(r'/\s*(m²|m2|cm²|cm2|nm²|nm2)\b', unit_str):
            if 'a' in unit_str:
                return 'current_per_area'
            if 'w' in unit_str:
                return 'power_per_area'
        
        # Time-based rates
        if re.search(r'/(s|sec|min|hr|h)\b', unit_str) or 'hz' in unit_str:
            return 'rate'
        
        return None

    def _infer_range_from_dimension(self, dimension: Optional[str]) -> Optional[Dict]:
        """
        Infer reasonable value ranges from physical dimension.
        """
        if not dimension:
            return None
        
        range_map = {
            'efficiency': {'min': 0, 'max': 100},
            'percentage': {'min': 0, 'max': 100},
            'quantum_yield': {'min': 0, 'max': 100},
            'temperature': {'min': 0.1, 'max': 5000},
            'efficiency': {'min': 0, 'max': 100},
            'voltage': {'min': -1e6, 'max': 1e6},
            'current': {'min': -1e6, 'max': 1e6},
            'frequency': {'min': 0.001, 'max': 1e15},
            'length': {'min': 1e-15, 'max': 1e9},
            'mass': {'min': 1e-18, 'max': 1e9},
            'time': {'min': 1e-15, 'max': 1e9},
            'pressure': {'min': 0, 'max': 1e12},
            'angle': {'min': 0, 'max': 360},
            'dimensionless': {'min': 0, 'max': 1e6},
        }
        
        # Per-mass quantities are typically positive
        if 'per_mass' in dimension or 'density' in dimension:
            return {'min': 0.0001, 'max': 1e9}
        
        # Concentrations
        if dimension == 'concentration':
            return {'min': 0, 'max': 1e6}
        
        return range_map.get(dimension)

    def _create_generic_schema(self) -> Dict[str, Any]:
        """
        Create minimal generic schema for unknown domains.
        """
        return {
            'value': {
                'units': [],
                'aliases': ['value', 'result', 'measurement', 'data'],
                'type': 'float',
                'dimension': None,
                'expected_range': None,
                'validation_rules': []
            },
            'concentration': {
                'units': ['M', 'mM', 'μM', 'mg/mL', 'g/L', '%'],
                'aliases': ['concentration', 'conc', 'amount'],
                'type': 'float',
                'dimension': 'concentration',
                'expected_range': {'min': 0, 'max': 1e6},
                'validation_rules': ['positive']
            },
            'size': {
                'units': ['m', 'cm', 'mm', 'μm', 'nm', 'Å'],
                'aliases': ['size', 'dimension', 'length', 'diameter', 'width', 'height'],
                'type': 'float',
                'dimension': 'length',
                'expected_range': {'min': 1e-12, 'max': 1e6},
                'validation_rules': ['positive']
            },
            'time_duration': {
                'units': ['s', 'min', 'hr', 'day', 'week', 'month', 'year'],
                'aliases': ['time', 'duration', 'period', 'interval'],
                'type': 'float',
                'dimension': 'time',
                'expected_range': {'min': 0, 'max': 1e9},
                'validation_rules': ['positive']
            },
            'temperature': {
                'units': ['K', '°C', '°F'],
                'aliases': ['temperature', 'temp', 'T'],
                'type': 'float',
                'dimension': 'temperature',
                'expected_range': {'min': 0, 'max': 5000},
                'validation_rules': []
            }
        }
    
    def create_default_energy_schema(self) -> Dict[str, Any]:
        """Create default schema for energy storage domain (supercapacitors/batteries)."""
        schema = {
            'attributes': {
                'specific_capacitance': {
                    'units': ['F/g', 'F/cm²', 'mF/g', 'mF/cm²'],
                    'expected_range': {'min': 0.1, 'max': 10000},
                    'type': 'float',
                    'aliases': [
                        'gravimetric capacitance', 'specific capacitance', 'Cs',
                        'specific capacitance (F g-1)', 'capacitance (F/g)'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'capacitance_per_mass'
                },
                'areal_capacitance': {
                    'units': ['F/cm²', 'mF/cm²', 'µF/cm²'],
                    'expected_range': {'min': 0.001, 'max': 100},
                    'type': 'float',
                    'aliases': [
                        'areal capacitance', 'capacitance (F cm-2)',
                        'specific areal capacitance'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'capacitance_per_area'
                },
                'energy_density': {
                    'units': ['Wh/kg', 'Wh/L', 'mWh/cm³', 'Wh cm-3', 'Wh kg-1'],
                    'expected_range': {'min': 0.001, 'max': 1000},
                    'type': 'float',
                    'aliases': [
                        'energy density', 'ED', 'specific energy',
                        'energy density (Wh kg-1)', 'gravimetric energy density'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'energy_per_mass'
                },
                'power_density': {
                    'units': ['W/kg', 'kW/kg', 'W/L', 'W cm-3', 'W kg-1'],
                    'expected_range': {'min': 0.01, 'max': 100000},
                    'type': 'float',
                    'aliases': [
                        'power density', 'PD', 'specific power',
                        'power density (W kg-1)', 'gravimetric power density'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'power_per_mass'
                },
                'current_density': {
                    'units': ['A/g', 'A/cm²', 'mA/g', 'mA/cm²', 'A kg-1'],
                    'expected_range': {'min': 0.0001, 'max': 100},
                    'type': 'float',
                    'aliases': [
                        'current density', 'j', 'J',
                        'current density (A g-1)', 'current density (A/g)'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'current_per_mass'
                },
                'scan_rate': {
                    'units': ['mV/s', 'V/s', 'mV s-1'],
                    'expected_range': {'min': 0.01, 'max': 10000},
                    'type': 'float',
                    'aliases': [
                        'scan rate', 'scanrate', 'sweep rate',
                        'potential scan rate', 'scan rate (mV s-1)'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'voltage_per_time'
                },
                'potential_window': {
                    'units': ['V', 'mV'],
                    'expected_range': {'min': 0.01, 'max': 10},
                    'type': 'float',
                    'aliases': [
                        'potential window', 'voltage window', 'working voltage',
                        'potential range', 'operating voltage'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'voltage'
                },
                'specific_surface_area': {
                    'units': ['m²/g', 'm2/g', 'm² g-1', 'cm²/g'],
                    'expected_range': {'min': 0.1, 'max': 5000},
                    'type': 'float',
                    'aliases': [
                        'specific surface area', 'SSA', 'BET surface area',
                        'surface area', 'BET'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'area_per_mass'
                },
                'pore_size': {
                    'units': ['nm', 'Å', 'um', 'µm', 'micrometer', 'angstrom'],
                    'expected_range': {'min': 0.1, 'max': 10000},
                    'type': 'float',
                    'aliases': [
                        'pore size', 'average pore size', 'pore diameter',
                        'pore width', 'mean pore size'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'length'
                },
                'pore_volume': {
                    'units': ['cm³/g', 'cm3/g', 'cc/g', 'mL/g'],
                    'expected_range': {'min': 0.001, 'max': 5},
                    'type': 'float',
                    'aliases': [
                        'pore volume', 'total pore volume',
                        'specific pore volume', 'pore volume (cm3 g-1)'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'volume_per_mass'
                },
                'micropore_volume': {
                    'units': ['cm³/g', 'cm3/g', 'cc/g'],
                    'expected_range': {'min': 0.0001, 'max': 2},
                    'type': 'float',
                    'aliases': [
                        'micropore volume', 'micro-pore volume',
                        'Vmicro', 'micropore volume (cm3 g-1)'
                    ],
                    'validation_rules': ['positive', 'unit_required'],
                    'dimension': 'volume_per_mass'
                },
                'electrolyte_type': {
                    'units': [],  # Categorical, no units
                    'expected_range': None,
                    'type': 'string',
                    'aliases': [
                        'electrolyte', 'electrolyte type', 'electrolyte used',
                        'ionic liquid', 'aqueous electrolyte', 'organic electrolyte'
                    ],
                    'validation_rules': ['not_numeric'],  # Should be text like "KOH", "H2SO4"
                    'dimension': None
                },
                'carbon_content': {
                    'units': ['%', 'wt%', 'at%'],
                    'expected_range': {'min': 0, 'max': 100},
                    'type': 'float',
                    'aliases': [
                        'carbon content', 'C content', 'carbon wt%',
                        'C wt%', 'mass fraction of carbon'
                    ],
                    'validation_rules': ['range_0_100', 'unit_required'],
                    'dimension': 'percentage'
                },
                'nitrogen_content': {
                    'units': ['%', 'wt%', 'at%'],
                    'expected_range': {'min': 0, 'max': 50},
                    'type': 'float',
                    'aliases': [
                        'nitrogen content', 'N content', 'nitrogen wt%',
                        'N wt%', 'mass fraction of nitrogen'
                    ],
                    'validation_rules': ['range_0_100', 'unit_required'],
                    'dimension': 'percentage'
                },
                'oxygen_content': {
                    'units': ['%', 'wt%', 'at%'],
                    'expected_range': {'min': 0, 'max': 50},
                    'type': 'float',
                    'aliases': [
                        'oxygen content', 'O content', 'oxygen wt%',
                        'O wt%', 'mass fraction of oxygen'
                    ],
                    'validation_rules': ['range_0_100', 'unit_required'],
                    'dimension': 'percentage'
                },
                'id_ig_ratio': {
                    'units': [],  # Dimensionless ratio
                    'expected_range': {'min': 0.1, 'max': 5},
                    'type': 'float',
                    'aliases': [
                        'ID/IG ratio', 'ID/IG', 'D/G ratio',
                        'I(D)/I(G)', 'Raman D/G ratio'
                    ],
                    'validation_rules': ['positive'],
                    'dimension': 'dimensionless'
                }
            },
            'unit_ontology': {
                'dimensions': {
                    'capacitance_per_mass': {
                        'base_unit': 'F/g',
                        'si_unit': 'F/kg',
                        'aliases': ['F g-1', 'F/g', 'F g−1', 'F/g⁻¹', 'Farad/g', 'mF/g', 'µF/g'],
                        'conversion': {
                            'F/kg': 1000,
                            'mF/g': 0.001,
                            'µF/g': 1e-6,
                            'uF/g': 1e-6,
                            'F/t': 1e-6
                        }
                    },
                    'energy_per_mass': {
                        'base_unit': 'Wh/kg',
                        'si_unit': 'J/kg',
                        'aliases': ['Wh kg-1', 'Wh/kg', 'kWh/kg', 'Wh kg−1', 'Wh/kg⁻¹'],
                        'conversion': {
                            'J/kg': 3.6,
                            'kJ/kg': 3600,
                            'kWh/kg': 1000
                        }
                    },
                    'power_per_mass': {
                        'base_unit': 'W/kg',
                        'si_unit': 'W/kg',
                        'aliases': ['W kg-1', 'W/kg', 'kW/kg', 'W g-1'],
                        'conversion': {
                            'kW/kg': 1000,
                            'W/g': 1000,
                            'mW/g': 1
                        }
                    },
                    'current_per_mass': {
                        'base_unit': 'A/g',
                        'si_unit': 'A/kg',
                        'aliases': ['A g-1', 'A/g', 'A kg-1', 'mA/g', 'mA mg-1'],
                        'conversion': {
                            'A/kg': 1000,
                            'mA/g': 0.001,
                            'mA/mg': 1
                        }
                    },
                    'voltage_per_time': {
                        'base_unit': 'V/s',
                        'si_unit': 'V/s',
                        'aliases': ['mV/s', 'mV s-1', 'V/s', 'mV/s⁻¹'],
                        'conversion': {
                            'mV/s': 0.001,
                            'V/ms': 1000
                        }
                    },
                    'voltage': {
                        'base_unit': 'V',
                        'si_unit': 'V',
                        'aliases': ['V', 'mV', 'kV', 'volts'],
                        'conversion': {
                            'mV': 0.001,
                            'kV': 1000
                        }
                    },
                    'area_per_mass': {
                        'base_unit': 'm²/g',
                        'si_unit': 'm²/kg',
                        'aliases': ['m2 g-1', 'm²/g', 'm² g−1', 'm²/g⁻¹', 'cm²/g'],
                        'conversion': {
                            'm²/kg': 1000,
                            'cm²/g': 0.0001,
                            'm²/t': 1
                        }
                    },
                    'length': {
                        'base_unit': 'nm',
                        'si_unit': 'm',
                        'aliases': ['nm', 'Å', 'um', 'µm', 'micrometer', 'angstrom', 'pm'],
                        'conversion': {
                            'Å': 0.1,
                            'um': 1000,
                            'µm': 1000,
                            'pm': 0.001,
                            'm': 1e9,
                            'mm': 1e6
                        }
                    },
                    'volume_per_mass': {
                        'base_unit': 'cm³/g',
                        'si_unit': 'm³/kg',
                        'aliases': ['cm3 g-1', 'cm³/g', 'cc/g', 'mL/g', 'mL g-1'],
                        'conversion': {
                            'm³/kg': 1000,
                            'mL/g': 1,
                            'L/kg': 1
                        }
                    },
                    'percentage': {
                        'base_unit': '%',
                        'si_unit': 'fraction',
                        'aliases': ['%', 'wt%', 'at%', 'mass%', 'atom%', 'volume%'],
                        'conversion': {
                            'fraction': 0.01,
                            'wt%': 1,
                            'at%': 1
                        }
                    },
                    'dimensionless': {
                        'base_unit': '',
                        'si_unit': '',
                        'aliases': ['', '-', 'ratio'],
                        'conversion': {}
                    }
                }
            }
        }
        
        self.schema = schema['attributes']
        self.unit_ontology = schema['unit_ontology']
        self._build_attribute_index()
        return schema
    
    def _build_attribute_index(self) -> None:
        """Build index of all aliases mapping to canonical attribute names."""
        self.attribute_index = {}
        for attr_name, attr_data in self.schema.items():
            # Index canonical name
            self.attribute_index[attr_name.lower()] = attr_name
            self.attribute_index[attr_name.lower().replace('_', ' ')] = attr_name
            # Index all aliases
            for alias in attr_data.get('aliases', []):
                self.attribute_index[alias.lower()] = attr_name
                # Also index without special chars
                clean_alias = re.sub(r'[^a-z0-9\s]', ' ', alias.lower())
                self.attribute_index[clean_alias] = attr_name
    
    def find_attribute_by_name(self, name: str) -> Optional[Tuple[str, Dict]]:
        """
        Find attribute by name or alias.
        Returns (canonical_name, attribute_data) or None.
        """
        name_lower = name.lower().strip()
        
        # Direct match
        if name_lower in self.attribute_index:
            canonical = self.attribute_index[name_lower]
            return canonical, self.schema.get(canonical, {})
        
        # Try with cleaned name
        clean_name = re.sub(r'[^a-z0-9\s]', ' ', name_lower)
        if clean_name in self.attribute_index:
            canonical = self.attribute_index[clean_name]
            return canonical, self.schema.get(canonical, {})
        
        # Partial word matching
        name_words = set(w for w in clean_name.split() if len(w) > 3)
        best_match = None
        best_score = 0
        
        for alias, canonical in self.attribute_index.items():
            alias_words = set(w for w in alias.split() if len(w) > 3)
            if alias_words:
                score = len(name_words & alias_words) / len(alias_words)
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_match = canonical
        
        if best_match:
            return best_match, self.schema.get(best_match, {})
        
        return None
    
    def get_attribute_units(self, attr_name: str) -> List[str]:
        """Get valid units for an attribute."""
        attr_data = self.schema.get(attr_name, {})
        return attr_data.get('units', [])
    
    def get_attribute_dimension(self, attr_name: str) -> Optional[str]:
        """Get physical dimension of attribute (e.g., 'energy_per_mass')."""
        attr_data = self.schema.get(attr_name, {})
        return attr_data.get('dimension')
    
    def get_expected_range(self, attr_name: str) -> Optional[Dict]:
        """Get expected value range for attribute."""
        attr_data = self.schema.get(attr_name, {})
        return attr_data.get('expected_range')
    
    def validate_value_type(self, attr_name: str, value: Any) -> bool:
        """Check if value type matches attribute expectation."""
        attr_data = self.schema.get(attr_name, {})
        expected_type = attr_data.get('type', 'float')
        
        if expected_type == 'float':
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'int':
            try:
                int(value)
                return True
            except (ValueError, TypeError):
                return False
        
        return True
    
    def get_dimension_for_unit(self, unit: str) -> Optional[str]:
        """Get physical dimension for a given unit."""
        unit_lower = unit.lower().strip()
        
        for dim_name, dim_data in self.unit_ontology.get('dimensions', {}).items():
            # Check base unit
            if unit_lower == dim_data['base_unit'].lower():
                return dim_name
            # Check aliases
            for alias in dim_data.get('aliases', []):
                if unit_lower == alias.lower():
                    return dim_name
                # Handle superscript variants
                alias_normalized = alias.lower().replace('−', '-').replace('⁻', '-')
                if unit_lower == alias_normalized:
                    return dim_name
        
        return None
    
    def are_units_compatible(self, unit1: str, unit2: str) -> bool:
        """Check if two units have the same physical dimension."""
        dim1 = self.get_dimension_for_unit(unit1)
        dim2 = self.get_dimension_for_unit(unit2)
        return dim1 is not None and dim1 == dim2
    
    def convert_to_base_unit(self, value: float, from_unit: str) -> Tuple[float, str]:
        """
        Convert value to base unit for its dimension.
        Returns (converted_value, base_unit).
        """
        dim = self.get_dimension_for_unit(from_unit)
        if not dim:
            return value, from_unit
        
        dim_data = self.unit_ontology['dimensions'].get(dim, {})
        conversions = dim_data.get('conversion', {})
        base_unit = dim_data.get('base_unit', from_unit)
        
        # Normalize unit string
        from_unit_norm = from_unit.lower().replace('−', '-').replace('⁻', '-')
        
        # Find conversion factor
        factor = conversions.get(from_unit_norm)
        if factor:
            return value * factor, base_unit
        
        # Try with common variants
        for unit_key, conv_factor in conversions.items():
            if from_unit_norm in unit_key or unit_key in from_unit_norm:
                return value * conv_factor, base_unit
        
        return value, from_unit


# Singleton instance for global use
_global_schema_loader: Optional[SchemaLoader] = None


def get_schema_loader(schema_path: Optional[str] = None) -> SchemaLoader:
    """Get or create global schema loader instance."""
    global _global_schema_loader
    if _global_schema_loader is None:
        _global_schema_loader = SchemaLoader(schema_path)
        if not schema_path:
            _global_schema_loader.create_default_energy_schema()
    return _global_schema_loader


def load_domain_schema(workspace: Optional[str] = None) -> SchemaLoader:
    """
    Load or create schema for ANY scientific domain.
    Supports: solar cells, magnetics, batteries, autism, biomedical, chemistry, etc.
    """
    if workspace:
        # Look for schema file in workspace
        schema_paths = [
            os.path.join(workspace, 'schema.json'),
            os.path.join(workspace, 'domain_schema.json'),
        ]
        
        for path in schema_paths:
            if os.path.exists(path):
                return SchemaLoader(path, workspace)
        
        # No schema file - create dynamically from aliases or use generic
        loader = SchemaLoader(workspace=workspace)
        loader.load_or_create_domain_schema(workspace)
        return loader
    else:
        # Fallback to generic loader if no workspace provided
        return SchemaLoader()



if __name__ == '__main__':
    # Test the schema loader
    loader = SchemaLoader()
    schema = loader.create_default_energy_schema()
    
    print("Schema loaded with attributes:")
    for attr in loader.schema.keys():
        print(f"  - {attr}")
    
    # Test attribute lookup
    test_names = [
        'specific capacitance',
        'energy density (Wh kg-1)',
        'SSA',
        'pore diameter',
        'scanrate'
    ]
    
    print("\nAttribute lookup tests:")
    for name in test_names:
        result = loader.find_attribute_by_name(name)
        if result:
            print(f"  '{name}' -> {result[0]}")
        else:
            print(f"  '{name}' -> NOT FOUND")
    
    # Test unit compatibility
    print("\nUnit compatibility tests:")
    print(f"  F/g vs mF/g: {loader.are_units_compatible('F/g', 'mF/g')}")
    print(f"  Wh/kg vs W/kg: {loader.are_units_compatible('Wh/kg', 'W/kg')}")
    print(f"  nm vs m²/g: {loader.are_units_compatible('nm', 'm²/g')}")
