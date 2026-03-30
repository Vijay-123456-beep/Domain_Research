"""
Validation Engine Module
Multi-layer validation system for scientific data extraction.
Implements unit, range, and context validation with confidence scoring.
"""
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from schema_loader import SchemaLoader, get_schema_loader
from unit_parser import UnitParser, get_unit_parser, ParsedValue


class ValidationStatus(Enum):
    """Validation result status."""
    VALID = "valid"
    INVALID_UNIT = "invalid_unit"
    INVALID_RANGE = "invalid_range"
    INVALID_TYPE = "invalid_type"
    INVALID_CONTEXT = "invalid_context"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


@dataclass
class ValidationResult:
    """Result of validating a datapoint."""
    status: ValidationStatus
    confidence: float  # 0.0 to 1.0
    value: Any
    unit: Optional[str]
    attribute: str
    reasons: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def is_acceptable(self, threshold: float = 0.5) -> bool:
        """Check if result is acceptable given confidence threshold."""
        return self.confidence >= threshold and self.status not in [
            ValidationStatus.REJECTED,
            ValidationStatus.INVALID_UNIT
        ]


class ValidationEngine:
    """
    Multi-layer validation system:
    1. Unit validation (highest priority)
    2. Type validation (numeric vs string)
    3. Range validation (expected min/max)
    4. Context validation (not header/caption/reference)
    """
    
    def __init__(self, schema_loader: Optional[SchemaLoader] = None, 
                 unit_parser: Optional[UnitParser] = None,
                 veritas_mode: bool = True):
        self.schema = schema_loader or get_schema_loader()
        self.unit_parser = unit_parser or get_unit_parser()
        self.veritas_mode = veritas_mode
    
    def validate_datapoint(self,
                          value: Any,
                          unit: Optional[str],
                          attribute: str,
                          context: Optional[Dict] = None) -> ValidationResult:
        """
        Run full validation pipeline on a datapoint.
        
        Args:
            value: The extracted value (numeric or string)
            unit: Detected unit (if any)
            attribute: Target attribute name
            context: Additional context (source type, surrounding text, etc.)
        
        Returns:
            ValidationResult with status and confidence score
        """
        context = context or {}
        reasons = []
        confidence = 1.0
        
        # Get schema for attribute
        schema_attr = self.schema.schema.get(attribute)
        if not schema_attr:
            # Try to find by alias
            found = self.schema.find_attribute_by_name(attribute)
            if found:
                attribute = found[0]
                schema_attr = found[1]
            else:
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    confidence=0.0,
                    value=value,
                    unit=unit,
                    attribute=attribute,
                    reasons=[f"Attribute '{attribute}' not found in schema"]
                )
        
        # === LAYER 1: Type Validation ===
        type_result = self._validate_type(value, schema_attr)
        if type_result['status'] == 'rejected':
            return ValidationResult(
                status=ValidationStatus.INVALID_TYPE,
                confidence=0.0,
                value=value,
                unit=unit,
                attribute=attribute,
                reasons=[type_result['reason']]
            )
        
        if type_result['status'] == 'penalty':
            confidence *= 0.5
            reasons.append(type_result['reason'])
        
        # === LAYER 2: Unit Validation (highest weight) ===
        unit_result = self._validate_unit(unit, schema_attr, value)
        
        if unit_result['status'] == 'rejected':
            return ValidationResult(
                status=ValidationStatus.INVALID_UNIT,
                confidence=0.0,
                value=value,
                unit=unit,
                attribute=attribute,
                reasons=[unit_result['reason']]
            )
        
        # Unit match score (0.0 to 1.0)
        unit_confidence = unit_result.get('confidence', 0.0)
        confidence *= (0.4 + 0.6 * unit_confidence)  # Unit is 60% of confidence
        
        if unit_result['status'] == 'warning':
            reasons.append(unit_result['reason'])
        
        # Use normalized unit
        final_unit = unit_result.get('normalized_unit', unit)
        
        # === LAYER 3: Range Validation ===
        range_result = self._validate_range(value, schema_attr)
        
        if range_result['status'] == 'rejected':
            return ValidationResult(
                status=ValidationStatus.INVALID_RANGE,
                confidence=0.1,
                value=value,
                unit=final_unit,
                attribute=attribute,
                reasons=[range_result['reason']]
            )
        
        if range_result['status'] == 'warning':
            confidence *= 0.7
            reasons.append(range_result['reason'])
        elif range_result['status'] == 'good':
            confidence = min(1.0, confidence + 0.1)  # Bonus for good range fit
        
        # === LAYER 4: Context Validation ===
        context_result = self._validate_context(context, schema_attr)
        
        if context_result['status'] == 'rejected':
            return ValidationResult(
                status=ValidationStatus.INVALID_CONTEXT,
                confidence=0.2,
                value=value,
                unit=final_unit,
                attribute=attribute,
                reasons=[context_result['reason']]
            )
        
        if context_result['status'] == 'warning':
            confidence *= 0.6
            reasons.append(context_result['reason'])
        
        # Determine final status
        if confidence >= 0.9 and status != ValidationStatus.REJECTED:
            status = ValidationStatus.VALID
        elif confidence >= 0.7:
            status = ValidationStatus.AMBIGUOUS
        else:
            status = ValidationStatus.REJECTED
            
        # VERITAS MODE OVERRIDE: Zero tolerance for any invalidity (Fix 3.2)
        if self.veritas_mode:
            if status in [ValidationStatus.AMBIGUOUS, ValidationStatus.INVALID_RANGE, ValidationStatus.INVALID_UNIT]:
                status = ValidationStatus.REJECTED
                confidence = 0.0
                reasons.append("[VERITAS] Rejected due to ambiguity or slight validation failure.")

        return ValidationResult(
            status=status,
            confidence=round(confidence, 3),
            value=value,
            unit=final_unit,
            attribute=attribute,
            reasons=reasons,
            suggestions=unit_result.get('suggestions', [])
        )
    
    def _validate_type(self, value: Any, schema_attr: Dict) -> Dict:
        """
        Validate that value type matches schema expectation.
        """
        expected_type = schema_attr.get('type', 'float')
        validation_rules = schema_attr.get('validation_rules', [])
        
        # Check if value is string that should be numeric
        if isinstance(value, str):
            # Try to parse as numeric
            try:
                float(value)
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False
            
            if expected_type in ['float', 'int'] and not is_numeric:
                # String value for numeric attribute - reject if it's long text
                if len(value) > 20 or 'not_numeric' in validation_rules:
                    return {
                        'status': 'rejected',
                        'reason': f"String value '{value[:30]}...' for numeric attribute"
                    }
            
            # Check for header-like content
            if self._is_likely_header(value):
                return {
                    'status': 'rejected',
                    'reason': f"Value appears to be table header: '{value[:30]}...'"
                }
            
            # Check for caption-like content
            if self._is_likely_caption(value):
                return {
                    'status': 'rejected',
                    'reason': f"Value appears to be figure caption: '{value[:30]}...'"
                }
        
        # Check for numeric-only attributes
        if 'not_numeric' in validation_rules and isinstance(value, (int, float)):
            # This is a categorical attribute, shouldn't have numeric value
            if value not in [0, 1]:  # Allow binary indicators
                return {
                    'status': 'penalty',
                    'reason': f"Numeric value {value} for categorical attribute"
                }
        
        return {'status': 'ok'}
    
    def _validate_unit(self, unit: Optional[str], schema_attr: Dict, value: Any) -> Dict:
        """
        Validate unit compatibility with attribute.
        Returns detailed result with confidence score.
        """
        expected_units = schema_attr.get('units', [])
        dimension = schema_attr.get('dimension')
        
        if not unit:
            # No unit detected
            if expected_units:
                return {
                    'status': 'warning',
                    'reason': 'No unit detected, but units expected',
                    'confidence': 0.2
                }
            else:
                # Categorical attribute, no unit needed
                return {
                    'status': 'ok',
                    'confidence': 0.5  # Neutral without unit
                }
        
        # Normalize unit
        normalized_unit = self.unit_parser._normalize_unit(unit)
        
        # Check exact match
        for expected in expected_units:
            if normalized_unit and normalized_unit.lower() == expected.lower():
                return {
                    'status': 'ok',
                    'confidence': 1.0,
                    'normalized_unit': expected
                }
            if self.unit_parser._normalize_unit(expected).lower() == normalized_unit.lower():
                return {
                    'status': 'ok',
                    'confidence': 1.0,
                    'normalized_unit': expected
                }
        
        # Check dimension compatibility
        if dimension and normalized_unit:
            value_dim = self.unit_parser.get_dimension(normalized_unit)
            if value_dim == dimension:
                return {
                    'status': 'ok',
                    'confidence': 0.7,  # Lower confidence for compatible but not exact
                    'normalized_unit': expected_units[0] if expected_units else unit,
                    'reason': f'Unit {unit} compatible with dimension {dimension}'
                }
        
        # Check if units are compatible via parser
        for expected in expected_units:
            if self.unit_parser.are_compatible(unit, expected):
                return {
                    'status': 'ok',
                    'confidence': 0.6,
                    'normalized_unit': expected
                }
        
        # Unit mismatch
        return {
            'status': 'rejected',
            'reason': f"Unit '{unit}' incompatible with expected units: {expected_units}",
            'confidence': 0.0,
            'suggestions': expected_units
        }
    
    def _validate_range(self, value: Any, schema_attr: Dict) -> Dict:
        """
        Validate value is within expected range.
        """
        expected_range = schema_attr.get('expected_range')
        if not expected_range:
            return {'status': 'ok'}
        
        # Try to convert to float
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return {'status': 'ok'}  # Can't validate range of non-numeric
        
        min_val = expected_range.get('min')
        max_val = expected_range.get('max')
        
        # Check bounds
        if min_val is not None and numeric_value < min_val:
            # How far below?
            if numeric_value < 0 and min_val >= 0:
                return {
                    'status': 'rejected',
                    'reason': f"Value {numeric_value} is negative but attribute expects positive values"
                }
            
            ratio = numeric_value / min_val if min_val != 0 else 0
            if ratio < 0.1:  # More than 10x below minimum
                return {
                    'status': 'rejected',
                    'reason': f"Value {numeric_value} is far below minimum {min_val}"
                }
            
            return {
                'status': 'warning',
                'reason': f"Value {numeric_value} below expected minimum {min_val}"
            }
        
        if max_val is not None and numeric_value > max_val:
            ratio = numeric_value / max_val
            if ratio > 10:  # More than 10x above maximum
                return {
                    'status': 'rejected',
                    'reason': f"Value {numeric_value} is far above maximum {max_val}"
                }
            
            return {
                'status': 'warning',
                'reason': f"Value {numeric_value} above expected maximum {max_val}"
            }
        
        # Good fit within range
        return {'status': 'good'}
    
    def _validate_context(self, context: Dict, schema_attr: Dict) -> Dict:
        """
        Validate extraction context (not header, caption, reference, etc.)
        """
        source_type = context.get('source_type', 'unknown')
        surrounding_text = context.get('surrounding_text', '')
        cell_content = context.get('cell_content', '')
        
        # Check for header cells
        if source_type == 'table_cell':
            if self._is_likely_header(cell_content):
                return {
                    'status': 'rejected',
                    'reason': f"Cell appears to be table header: '{cell_content[:40]}...'"
                }
            
            # Check content length - reject very long cells
            if len(str(cell_content)) > 100:
                return {
                    'status': 'rejected',
                    'reason': f"Cell content too long ({len(str(cell_content))} chars), likely merged cells"
                }
        
        # Check for figure captions
        if source_type == 'text' and self._is_likely_caption(surrounding_text):
            return {
                'status': 'warning',
                'reason': "Text appears to be figure caption, not data"
            }
        
        # Check for reference numbers
        if isinstance(cell_content, str):
            # Pattern: standalone numbers like [1], [23], (4)
            if re.match(r'^\s*[\[(]\s*\d+\s*[\])]\s*$', cell_content):
                return {
                    'status': 'rejected',
                    'reason': f"Value appears to be reference number: '{cell_content}'"
                }
        
        return {'status': 'ok'}
    
    def _is_likely_header(self, text: str) -> bool:
        """Check if text is likely a table header."""
        if not text:
            return False
        
        text_str = str(text).strip()
        
        # Long text is likely not a data value
        if len(text_str) > 30:
            return True
        
        # Contains only text (no numbers)
        if not re.search(r'\d', text_str):
            return True
        
        # Common header keywords
        header_keywords = [
            'sample', 'material', 'electrode', 'condition', 'parameter',
            'property', 'value', 'unit', 'reference', 'note', 'remark'
        ]
        text_lower = text_str.lower()
        for keyword in header_keywords:
            if keyword in text_lower and len(text_str) < 50:
                return True
        
        return False
    
    def _is_likely_caption(self, text: str) -> bool:
        """Check if text is likely a figure caption."""
        if not text:
            return False
        
        text_str = str(text).strip()
        
        # Starts with figure indicator
        if re.match(r'^(?:Fig\.?|Figure)\s*\d', text_str, re.IGNORECASE):
            return True
        
        # Very long descriptive text
        if len(text_str) > 100 and text_str.count('.') > 2:
            return True
        
        return False
    
    def batch_validate(self,
                      datapoints: List[Dict],
                      min_confidence: float = 0.5) -> Tuple[List[ValidationResult], List[ValidationResult]]:
        """
        Validate multiple datapoints, return (accepted, rejected).
        """
        accepted = []
        rejected = []
        
        for dp in datapoints:
            result = self.validate_datapoint(
                value=dp.get('value'),
                unit=dp.get('unit'),
                attribute=dp.get('attribute'),
                context=dp.get('context', {})
            )
            
            if result.is_acceptable(min_confidence):
                accepted.append(result)
            else:
                rejected.append(result)
        
        return accepted, rejected


# Global instance
_global_validation_engine: Optional[ValidationEngine] = None


def get_validation_engine(schema_loader: Optional[SchemaLoader] = None) -> ValidationEngine:
    """Get or create global validation engine."""
    global _global_validation_engine
    if _global_validation_engine is None:
        _global_validation_engine = ValidationEngine(schema_loader)
    return _global_validation_engine


if __name__ == '__main__':
    # Test validation engine
    engine = ValidationEngine()
    
    test_cases = [
        {
            'value': 150.5,
            'unit': 'F/g',
            'attribute': 'specific_capacitance',
            'context': {'source_type': 'table_cell', 'cell_content': '150.5'}
        },
        {
            'value': 'G-peak Position (cm-1)',
            'unit': None,
            'attribute': 'electrolyte_type',
            'context': {'source_type': 'table_cell', 'cell_content': 'G-peak Position (cm-1)'}
        },
        {
            'value': 10000,
            'unit': 'W/kg',
            'attribute': 'energy_density',
            'context': {'source_type': 'table_cell', 'cell_content': '10000'}
        },
        {
            'value': 0.5,
            'unit': 'nm',
            'attribute': 'pore_size',
            'context': {'source_type': 'table_cell', 'cell_content': '0.5'}
        },
    ]
    
    print("Validation tests:")
    for test in test_cases:
        result = engine.validate_datapoint(**test)
        print(f"\n  {test['value']} {test['unit']} -> {test['attribute']}")
        print(f"    Status: {result.status.value}, Confidence: {result.confidence}")
        if result.reasons:
            print(f"    Reasons: {result.reasons}")
