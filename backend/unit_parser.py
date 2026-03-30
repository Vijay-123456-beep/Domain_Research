"""
Unit Parser Module
Detects, normalizes, and validates units from scientific text and tables.
"""
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ParsedValue:
    """Represents a parsed numeric value with unit and metadata."""
    value: float
    raw_value: str
    unit: Optional[str]
    raw_unit: Optional[str]
    confidence: float
    value_type: str  # 'single', 'range', 'with_error', 'scientific'
    
    def __repr__(self):
        return f"ParsedValue({self.value} {self.unit}, conf={self.confidence:.2f})"


class UnitParser:
    """
    Parses units from scientific text with support for:
    - Standard unit symbols (F/g, Wh/kg, nm, etc.)
    - Unicode superscripts and subscripts
    - Range values (10-20, 10~20)
    - Scientific notation (1.5e-3, 1.5×10⁻³)
    - Parenthetical units (value (unit))
    """
    
    def __init__(self):
        # Common unit patterns by dimension
        self.unit_patterns = {
            'capacitance_per_mass': [
                r'F\s*g\s*[-−]\s*1', r'F/g', r'F\s*g\s*[-−]?¹',
                r'mF\s*g\s*[-−]\s*1', r'mF/g', r'mF\s*g\s*[-−]?¹',
                r'[µμ]F\s*g\s*[-−]\s*1', r'[µμ]F/g',
                r'F\s*cm\s*[-−]\s*2', r'F/cm²', r'mF\s*cm\s*[-−]\s*2'
            ],
            'energy_per_mass': [
                r'Wh\s*kg\s*[-−]\s*1', r'Wh/kg', r'Wh\s*kg\s*[-−]?¹',
                r'kWh\s*kg\s*[-−]\s*1', r'kWh/kg',
                r'Wh\s*L\s*[-−]\s*1', r'Wh/L',
                r'mWh\s*cm\s*[-−]\s*3', r'mWh/cm³'
            ],
            'power_per_mass': [
                r'W\s*kg\s*[-−]\s*1', r'W/kg', r'W\s*kg\s*[-−]?¹',
                r'kW\s*kg\s*[-−]\s*1', r'kW/kg',
                r'W\s*g\s*[-−]\s*1', r'W/g',
                r'mW\s*g\s*[-−]\s*1', r'mW/g'
            ],
            'current_per_mass': [
                r'A\s*g\s*[-−]\s*1', r'A/g', r'A\s*g\s*[-−]?¹',
                r'A\s*kg\s*[-−]\s*1', r'A/kg',
                r'mA\s*g\s*[-−]\s*1', r'mA/g',
                r'mA\s*mg\s*[-−]\s*1', r'mA/mg'
            ],
            'voltage_per_time': [
                r'mV\s*s\s*[-−]\s*1', r'mV/s', r'mV\s*s\s*[-−]?¹',
                r'V\s*s\s*[-−]\s*1', r'V/s',
                r'mV\s*s\s*[-−]?⁻¹'
            ],
            'voltage': [
                r'V\b', r'mV\b', r'kV\b', r'volts?'
            ],
            'area_per_mass': [
                r'm²\s*g\s*[-−]\s*1', r'm²/g', r'm2\s*g\s*[-−]\s*1',
                r'm²\s*kg\s*[-−]\s*1', r'm²/kg',
                r'cm²\s*g\s*[-−]\s*1', r'cm²/g', r'cm2/g'
            ],
            'length': [
                r'nm\b', r'[µμ]m\b', r'um\b', r'Å\b', r'angstrom',
                r'pm\b', r'mm\b', r'cm\b', r'm\b'
            ],
            'volume_per_mass': [
                r'cm³\s*g\s*[-−]\s*1', r'cm³/g', r'cm3/g',
                r'cc\s*/\s*g', r'cc/g',
                r'mL\s*g\s*[-−]\s*1', r'mL/g', r'mL\s*g\s*[-−]?¹'
            ],
            'percentage': [
                r'%', r'wt%', r'at%', r'mass%', r'atom%'
            ],
            'frequency': [
                r'Hz\b', r'kHz\b', r'MHz\b', r'GHz\b',
                r's\s*[-−]\s*1', r's[-−]¹', r'/s'
            ],
            'temperature': [
                r'°?C\b', r'°?F\b', r'K\b',
                r'°C', r'℃', r'°F'
            ],
            'time': [
                r's\b', r'sec\b', r'seconds?',
                r'min\b', r'minutes?',
                r'h\b', r'hr\b', r'hours?',
                r'd\b', r'days?'
            ],
            'mass': [
                r'g\b', r'kg\b', r'mg\b', r'[µμ]g\b', r'ug\b',
                r't\b', r'tons?'
            ]
        }
        
        # Compile master pattern for unit extraction
        self.unit_symbols = []
        for dim, patterns in self.unit_patterns.items():
            self.unit_symbols.extend(patterns)
        
        # Numeric patterns
        self.numeric_patterns = [
            # Scientific notation: 1.5e-3, 1.5×10⁻³, 1.5·10^(-3)
            r'[-+]?\d+\.?\d*[eE][-+]?\d+',
            r'[-+]?\d+\.?\d*\s*[×xX]\s*10\s*[-−]?\s*\d+',
            r'[-+]?\d+\.?\d*\s*[·•]\s*10\^?\s*[-−]?\s*\d+',
            # Standard decimal: 123.45, .45, 123
            r'[-+]?\d+\.\d+',
            r'[-+]?\.\d+',
            r'[-+]?\d+',
        ]
        
        # Range indicators
        self.range_separators = r'[-–—~∼≈]+'
    
    def extract_units_from_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract all (value, unit) pairs from text.
        Returns list of (value_str, unit_str).
        """
        results = []
        if not text:
            return results
        
        # Pattern: value (unit) or value [unit]
        paren_pattern = re.compile(
            r'(\d+\.?\d*)\s*' +
            r'[\(\[]([^\)\]]+)[\)\]]',
            re.IGNORECASE
        )
        
        for match in paren_pattern.finditer(text):
            value = match.group(1)
            unit = match.group(2).strip()
            results.append((value, unit))
        
        # Pattern: value unit (with space)
        # Look for numeric followed by unit pattern
        unit_regex = '|'.join(f'({p})' for p in self.unit_symbols)
        full_pattern = re.compile(
            r'(\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(' + unit_regex + r')',
            re.IGNORECASE
        )
        
        for match in full_pattern.finditer(text):
            value = match.group(1)
            unit = match.group(2)
            results.append((value, unit))
        
        return results
    
    def extract_unit_from_header(self, header_text: str) -> Optional[str]:
        """
        Extract unit from table header text.
        Headers often have format: "Attribute Name (unit)" or "Attribute [unit]"
        """
        if not header_text:
            return None
        
        # Pattern: (unit) or [unit] at end
        unit_patterns = [
            r'[\(\[]([^\)\]]+(?:/|g-|kg-|cm-|m-)[^\)\]]*)[\)\]]',  # (F/g), [Wh kg-1]
            r'[\(\[]([^\)\]]*(?:²|³|°|%)[^\)\]]*)[\)\]]',  # (m²/g), (℃), (%)
            r'[\(\[]([^\)\]]*(?:nm|μm|um|mm|cm|m|s|Hz)[^\)\]]*)[\)\]]',  # (nm), (Hz)
        ]
        
        for pattern in unit_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                unit = match.group(1).strip()
                # Clean up the unit
                unit = self._normalize_unit(unit)
                return unit
        
        return None
    
    def parse_value_string(self, value_str: str, unit_str: Optional[str] = None) -> Optional[ParsedValue]:
        """
        Parse a value string that may contain ranges, errors, etc.
        """
        if not value_str:
            return None
        
        value_str = str(value_str).strip()
        
        # Check for range: "10-20", "10~20", "10 – 20"
        range_match = re.match(
            rf'^\s*({self.numeric_patterns[0]})\s*{self.range_separators}\s*({self.numeric_patterns[0]})\s*$',
            value_str
        )
        
        if range_match:
            try:
                val1 = float(range_match.group(1))
                val2 = float(range_match.group(2))
                avg_val = (val1 + val2) / 2
                
                return ParsedValue(
                    value=avg_val,
                    raw_value=value_str,
                    unit=self._normalize_unit(unit_str) if unit_str else None,
                    raw_unit=unit_str,
                    confidence=0.7,  # Slightly lower for ranges
                    value_type='range'
                )
            except ValueError:
                pass
        
        # Check for value with error: "10.5 ± 0.3" or "10.5 +/- 0.3"
        error_match = re.match(
            rf'^\s*({self.numeric_patterns[0]})\s*[±\+\-/]+\s*({self.numeric_patterns[0]})\s*$',
            value_str
        )
        
        if error_match:
            try:
                val = float(error_match.group(1))
                
                return ParsedValue(
                    value=val,
                    raw_value=value_str,
                    unit=self._normalize_unit(unit_str) if unit_str else None,
                    raw_unit=unit_str,
                    confidence=0.8,
                    value_type='with_error'
                )
            except ValueError:
                pass
        
        # Single value
        single_match = re.match(
            rf'^\s*({self.numeric_patterns[0]})\s*$',
            value_str
        )
        
        if single_match:
            try:
                val = float(single_match.group(1))
                
                return ParsedValue(
                    value=val,
                    raw_value=value_str,
                    unit=self._normalize_unit(unit_str) if unit_str else None,
                    raw_unit=unit_str,
                    confidence=1.0,
                    value_type='single'
                )
            except ValueError:
                pass
        
        # Try to extract just the first numeric value from mixed text
        # Fixed regex: Must contain at least one digit!
        numeric_extract = re.search(r'(-?\d+\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+(?:[eE][-+]?\d+)?)', value_str)
        if numeric_extract:
            try:
                val = float(numeric_extract.group(1))
                
                return ParsedValue(
                    value=val,
                    raw_value=value_str,
                    unit=self._normalize_unit(unit_str) if unit_str else None,
                    raw_unit=unit_str,
                    confidence=0.4,  # Low confidence for extracted value
                    value_type='extracted'
                )
            except ValueError:
                pass
        
        return None
    
    def _normalize_unit(self, unit: Optional[str]) -> Optional[str]:
        """
        Normalize unit string to canonical form.
        """
        if not unit:
            return None
        
        unit = unit.strip()
        
        # Remove extra spaces
        unit = re.sub(r'\s+', ' ', unit)
        
        # Normalize superscript minus
        unit = unit.replace('−', '-').replace('⁻', '-')
        
        # Normalize superscript numbers
        superscript_map = {
            '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
            '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁰': '0'
        }
        for superscript, normal in superscript_map.items():
            unit = unit.replace(superscript, normal)
        
        # Handle various dash types
        unit = unit.replace('–', '-').replace('—', '-')
        
        # Normalize common forms
        normalizations = {
            # Capacitance
            'F g-1': 'F/g',
            'F g⁻¹': 'F/g',
            'F kg-1': 'F/kg',
            'mF g-1': 'mF/g',
            'uF/g': 'µF/g',
            'μF/g': 'µF/g',
            'F cm-2': 'F/cm²',
            
            # Energy
            'Wh kg-1': 'Wh/kg',
            'Wh kg⁻¹': 'Wh/kg',
            'Wh L-1': 'Wh/L',
            
            # Power
            'W kg-1': 'W/kg',
            'W kg⁻¹': 'W/kg',
            
            # Current
            'A g-1': 'A/g',
            'A g⁻¹': 'A/g',
            'mA g-1': 'mA/g',
            
            # Scan rate
            'mV s-1': 'mV/s',
            'mV s⁻¹': 'mV/s',
            
            # Surface area
            'm2 g-1': 'm²/g',
            'm² g-1': 'm²/g',
            'm² g⁻¹': 'm²/g',
            'm2/g': 'm²/g',
            'cm2/g': 'cm²/g',
            
            # Length
            'um': 'µm',
            'uM': 'µm',
            'micrometer': 'µm',
            'nanometer': 'nm',
            'angstrom': 'Å',
            
            # Volume
            'cm3 g-1': 'cm³/g',
            'cm³ g-1': 'cm³/g',
            'cc g-1': 'cc/g',
            'mL g-1': 'mL/g',
            
            # Percentage
            'wt%': '%',
            'at%': '%',
            'mass%': '%',
            'atom%': '%',
        }
        
        # Check for exact match
        if unit in normalizations:
            return normalizations[unit]
        
        # Check case-insensitive
        unit_lower = unit.lower()
        for orig, norm in normalizations.items():
            if unit_lower == orig.lower():
                return norm
        
        return unit
    
    def get_dimension(self, unit: str) -> Optional[str]:
        """
        Get physical dimension for a unit.
        """
        unit_normalized = self._normalize_unit(unit)
        if not unit_normalized:
            return None
        
        for dimension, patterns in self.unit_patterns.items():
            for pattern in patterns:
                # Create a simple match by removing regex special chars
                simple_pattern = pattern.replace('\\', '').replace('\b', '').replace('?', '').replace('*', '')
                if simple_pattern.lower() in unit_normalized.lower():
                    return dimension
        
        return None
    
    def are_compatible(self, unit1: str, unit2: str) -> bool:
        """Check if two units have the same physical dimension."""
        dim1 = self.get_dimension(unit1)
        dim2 = self.get_dimension(unit2)
        
        if dim1 and dim2:
            return dim1 == dim2
        
        return False
    
    def is_valid_unit_for_attribute(self, unit: str, attribute_units: List[str]) -> bool:
        """
        Check if a unit is valid for an attribute based on:
        1. Exact match
        2. Same dimension
        """
        if not unit:
            return False
        
        unit_normalized = self._normalize_unit(unit)
        
        # Check exact match
        for attr_unit in attribute_units:
            if unit_normalized.lower() == attr_unit.lower():
                return True
            if self._normalize_unit(attr_unit).lower() == unit_normalized.lower():
                return True
        
        # Check dimension compatibility
        unit_dim = self.get_dimension(unit_normalized)
        if unit_dim:
            for attr_unit in attribute_units:
                attr_dim = self.get_dimension(attr_unit)
                if attr_dim and unit_dim == attr_dim:
                    return True
        
        return False
    
    def extract_from_table_cell(self, cell_value: str, header_unit: Optional[str] = None) -> Optional[ParsedValue]:
        """
        Extract value and unit from a table cell.
        Cell may contain just value (unit in header) or value with unit.
        """
        if not cell_value:
            return None
        
        cell_str = str(cell_value).strip()
        
        # First try to find unit within the cell
        cell_units = self.extract_units_from_text(cell_str)
        
        if cell_units:
            # Cell contains its own unit
            value_str, unit_str = cell_units[0]
            return self.parse_value_string(value_str, unit_str)
        
        # No unit in cell, use header unit if provided
        if header_unit:
            # Try to extract just numeric value
            numeric_match = re.match(r'^\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*$', cell_str)
            if numeric_match:
                return self.parse_value_string(numeric_match.group(1), header_unit)
        
        # Try to parse anyway (might be numeric without unit)
        return self.parse_value_string(cell_str, header_unit)


# Global instance
_global_unit_parser: Optional[UnitParser] = None


def get_unit_parser() -> UnitParser:
    """Get or create global unit parser."""
    global _global_unit_parser
    if _global_unit_parser is None:
        _global_unit_parser = UnitParser()
    return _global_unit_parser


if __name__ == '__main__':
    # Test the unit parser
    parser = UnitParser()
    
    test_cases = [
        ("123.5 F/g", None),
        ("10-20", "mV/s"),
        ("1.5e-3", "A/g"),
        (" capacitance (F g⁻¹) ", None),
        ("10.5 ± 0.3", "nm"),
        ("G-peak Position (cm-1)", None),
    ]
    
    print("Unit parsing tests:")
    for text, default_unit in test_cases:
        result = parser.extract_from_table_cell(text, default_unit)
        if result:
            print(f"  '{text}' -> {result.value} {result.unit} (conf={result.confidence:.2f})")
        else:
            print(f"  '{text}' -> FAILED")
    
    # Test unit compatibility
    print("\nUnit compatibility tests:")
    print(f"  F/g vs mF/g: {parser.are_compatible('F/g', 'mF/g')}")
    print(f"  Wh/kg vs W/kg: {parser.are_compatible('Wh/kg', 'W/kg')}")
    print(f"  nm vs m²/g: {parser.are_compatible('nm', 'm²/g')}")
