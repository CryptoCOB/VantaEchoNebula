"""
Schema-Driven Evolution Integration

This module integrates all VoxSigil schemas (1.4, 1.5, 1.8-omega, SMART_MRAP) into 
the EVO/NAS pipeline to use them as structural DNA templates for:

1. EVO: Evolutionary optimization guided by schema constraints
2. NAS: Neural architecture search within schema frameworks  
3. BLT: Schema-aware compression and agent encoding
4. Nebula: Self-structuring using schema blueprints

The schemas serve as:
- Structural constraints for evolution
- Templates for agent creation
- Validation frameworks for generated architectures
- Cognitive scaffolding for self-improvement
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SchemaConstraint:
    """Represents a structural constraint from a VoxSigil schema"""
    schema_source: str  # Which schema it comes from
    constraint_type: str  # Type of constraint (structure, cognitive, embodiment, etc.)
    required_fields: List[str]
    optional_fields: List[str]
    validation_rules: Dict[str, Any]
    templates: Dict[str, Any]
    
class SchemaEvolutionIntegrator:
    """Integrates VoxSigil schemas with EVO/NAS systems for guided evolution"""
    
    def __init__(self, schema_directory: str = "training/schema"):
        self.schema_dir = Path(schema_directory)
        self.schemas = {}
        self.constraint_library = {}
        self.template_library = {}
        self.cognitive_primitives = {}
        
        # Load all schemas
        self.load_all_schemas()
        self.build_constraint_library()
        
    def load_all_schemas(self) -> None:
        """Load all VoxSigil schemas from the schema directory"""
        
        schema_files = {
            "voxsigil_1_4": "voxsigil-schema1.4-uni.yaml",
            "voxsigil_1_5": "voxsigil-schema-holo-1.5.yaml", 
            "voxsigil_1_8_omega": "voxsigil-1.8-holo-omega.json",
            "smart_mrap": "smart_mrap_template.yaml"
        }
        
        for schema_name, filename in schema_files.items():
            try:
                file_path = self.schema_dir / filename
                if file_path.exists():
                    if filename.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self.schemas[schema_name] = json.load(f)
                    else:  # YAML
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self.schemas[schema_name] = yaml.safe_load(f)
                    
                    logger.info(f"Loaded schema: {schema_name} from {filename}")
                else:
                    logger.warning(f"[WARNING] Schema file not found: {filename}")
                    
            except Exception as e:
                logger.error(f"[ERROR] Failed to load schema {schema_name}: {e}")
                
    def build_constraint_library(self) -> None:
        """Build a library of constraints from all schemas for EVO/NAS guidance"""
        
        # VoxSigil 1.4 - Foundational constraints
        if "voxsigil_1_4" in self.schemas:
            schema = self.schemas["voxsigil_1_4"]
            self._extract_constraints_v14(schema)
            
        # VoxSigil 1.5 - Advanced constraints  
        if "voxsigil_1_5" in self.schemas:
            schema = self.schemas["voxsigil_1_5"]
            self._extract_constraints_v15(schema)
            
        # VoxSigil 1.8 Omega - Agent creation constraints
        if "voxsigil_1_8_omega" in self.schemas:
            schema = self.schemas["voxsigil_1_8_omega"]
            self._extract_constraints_omega(schema)
            
        # SMART_MRAP - Cognitive principle constraints
        if "smart_mrap" in self.schemas:
            schema = self.schemas["smart_mrap"]
            self._extract_cognitive_constraints(schema)
            
        logger.info(f"Built constraint library with {len(self.constraint_library)} constraint types")
        
    def _extract_constraints_v14(self, schema: Dict[str, Any]) -> None:
        """Extract structural constraints from VoxSigil 1.4 schema"""
        
        if "voxsigil_schema_overview_1_4_alpha" in schema:
            sections = schema["voxsigil_schema_overview_1_4_alpha"]["sections"]
            
            for section_name, section_data in sections.items():
                if "fields" in section_data:
                    constraint = SchemaConstraint(
                        schema_source="voxsigil_1_4",
                        constraint_type=section_name,
                        required_fields=[],
                        optional_fields=[],
                        validation_rules={},
                        templates={}
                    )
                    
                    # Extract field requirements
                    for field_name, field_spec in section_data["fields"].items():
                        if isinstance(field_spec, dict) and field_spec.get("required", False):
                            constraint.required_fields.append(field_name)
                        else:
                            constraint.optional_fields.append(field_name)
                            
                        # Store field templates
                        constraint.templates[field_name] = field_spec
                        
                    self.constraint_library[f"v14_{section_name}"] = constraint
                    
    def _extract_constraints_v15(self, schema: Dict[str, Any]) -> None:
        """Extract advanced constraints from VoxSigil 1.5 holo-alpha schema"""
        
        if "properties" in schema:
            properties = schema["properties"]
            required = schema.get("required", [])
            
            # Group related properties into constraint categories
            constraint_categories = {
                "core_identification": ["sigil", "name", "alias", "tag", "tags"],
                "cognitive_modeling": ["embodiment_profile", "self_model", "learning_architecture_profile"],
                "sensory_experience": ["audio", "ya", "multi_sensory_profile"],
                "knowledge_representation": ["knowledge_representation_and_grounding"],
                "architectural_scaffolding": ["consciousness_scaffold_contribution_level", "cognitive_scaffold_role_in_vanta"]
            }
            
            for category, field_list in constraint_categories.items():
                constraint = SchemaConstraint(
                    schema_source="voxsigil_1_5",
                    constraint_type=category,
                    required_fields=[f for f in field_list if f in required],
                    optional_fields=[f for f in field_list if f not in required],
                    validation_rules={},
                    templates={}
                )
                
                # Extract templates for each field
                for field_name in field_list:
                    if field_name in properties:
                        constraint.templates[field_name] = properties[field_name]
                        
                self.constraint_library[f"v15_{category}"] = constraint
                
    def _extract_constraints_omega(self, schema: Dict[str, Any]) -> None:
        """Extract agent creation constraints from VoxSigil 1.8 omega schema"""
        
        # This is the complete agent template - extract its structure
        constraint = SchemaConstraint(
            schema_source="voxsigil_1_8_omega",
            constraint_type="complete_agent_architecture",
            required_fields=schema.get("required", []),  # Fix: use "required" not "required_fields"
            optional_fields=list(schema.get("properties", {}).keys()),  # All properties as optional
            validation_rules=schema.get("validation_rules", {}),
            templates=schema.get("properties", {})  # Use properties as templates
        )
        
        # Extract cognitive architecture constraints
        if "cognitive_architecture" in schema:
            cognitive_arch = schema["cognitive_architecture"]
            constraint.templates["cognitive_architecture"] = cognitive_arch
            
        # Extract operational constraints  
        if "operational_parameters" in schema:
            ops_params = schema["operational_parameters"]
            constraint.templates["operational_parameters"] = ops_params
            
        self.constraint_library["omega_complete_agent"] = constraint
        
    def _extract_cognitive_constraints(self, schema: Dict[str, Any]) -> None:
        """Extract cognitive principle constraints from SMART_MRAP schema"""
        
        if "SMART_MRAP" in schema:
            smart_mrap = schema["SMART_MRAP"]
            
            constraint = SchemaConstraint(
                schema_source="smart_mrap",
                constraint_type="cognitive_principles",
                required_fields=["Specific", "Measurable", "Achievable", "Relevant", "Transferable"],
                optional_fields=[],
                validation_rules={},
                templates={"SMART_MRAP": smart_mrap}
            )
            
            self.constraint_library["cognitive_principles"] = constraint
            
    def get_evo_constraints(self, evolution_target: str) -> Dict[str, Any]:
        """Get schema constraints for EVO (Evolutionary Optimizer) systems"""
        
        evo_constraints = {
            "structural_requirements": [],
            "cognitive_requirements": [],
            "validation_templates": {},
            "fitness_functions": []
        }
        
        # Add foundational structural requirements from VoxSigil 1.4
        if "v14_1_core_identification_and_classification" in self.constraint_library:
            constraint = self.constraint_library["v14_1_core_identification_and_classification"]
            evo_constraints["structural_requirements"].extend(constraint.required_fields)
            evo_constraints["validation_templates"].update(constraint.templates)
            
        # Add cognitive modeling requirements from VoxSigil 1.5
        if "v15_cognitive_modeling" in self.constraint_library:
            constraint = self.constraint_library["v15_cognitive_modeling"]
            evo_constraints["cognitive_requirements"].extend(constraint.required_fields)
            evo_constraints["validation_templates"].update(constraint.templates)
            
        # Add SMART_MRAP fitness criteria
        if "cognitive_principles" in self.constraint_library:
            constraint = self.constraint_library["cognitive_principles"]
            evo_constraints["fitness_functions"] = list(constraint.required_fields)
            
        logger.info(f"Generated EVO constraints for {evolution_target}")
        return evo_constraints
        
    def get_nas_constraints(self, architecture_target: str) -> Dict[str, Any]:
        """Get schema constraints for NAS (Neural Architecture Search) systems"""
        
        nas_constraints = {
            "architecture_templates": {},
            "layer_specifications": {},
            "connection_patterns": {},
            "cognitive_scaffolds": []
        }
        
        # Use VoxSigil 1.5 architectural scaffolding as NAS templates
        if "v15_architectural_scaffolding" in self.constraint_library:
            constraint = self.constraint_library["v15_architectural_scaffolding"]
            nas_constraints["architecture_templates"].update(constraint.templates)
            
        # Use omega schema for complete agent architecture patterns
        if "omega_complete_agent" in self.constraint_library:
            constraint = self.constraint_library["omega_complete_agent"]
            if "cognitive_architecture" in constraint.templates:
                nas_constraints["layer_specifications"] = constraint.templates["cognitive_architecture"]
                
        logger.info(f"Generated NAS constraints for {architecture_target}")
        return nas_constraints
        
    def get_blt_encoding_templates(self) -> Dict[str, Any]:
        """Get schema templates for BLT encoding of evolved agents"""
        
        encoding_templates = {
            "sigil_structure": {},
            "compression_patterns": {},
            "expansion_rules": {}
        }
        
        # Use all schemas to create encoding templates
        for schema_name, constraint in self.constraint_library.items():
            encoding_templates["sigil_structure"][schema_name] = {
                "required": constraint.required_fields,
                "optional": constraint.optional_fields,
                "templates": constraint.templates
            }
            
        # Add SMART_MRAP as compression pattern
        if "cognitive_principles" in self.constraint_library:
            constraint = self.constraint_library["cognitive_principles"]
            encoding_templates["compression_patterns"]["cognitive_principles"] = constraint.templates
            
        logger.info("🜹 Generated BLT encoding templates from all schemas")
        return encoding_templates
        
    def validate_evolved_architecture(self, architecture: Dict[str, Any], 
                                    target_schema: str = "voxsigil_1_8_omega") -> Dict[str, Any]:
        """Validate an evolved architecture against schema constraints"""
        
        # Check if this is a raw neural architecture vs a complete agent spec
        is_neural_arch = self._is_neural_architecture(architecture)
        
        if is_neural_arch:
            return self._validate_neural_architecture(architecture, target_schema)
        else:
            return self._validate_agent_architecture(architecture, target_schema)
    
    def _is_neural_architecture(self, architecture: Dict[str, Any]) -> bool:
        """Check if this is a raw neural network architecture"""
        neural_keys = ['layers', 'input_dim', 'output_dim', 'activations', 'device']
        agent_keys = ['core_identification', 'cognitive_modeling', 'sensory_experience']
        voxsigil_keys = ['sigil', 'name', 'principle', 'usage', 'SMART_MRAP']  # Voxsigil agent keys
        
        # Enhanced genome keys (from ArchitectureGenome with VoxSIGIL fields)
        enhanced_genome_keys = [
            'sigil', 'name', 'principle',  # Core identity
            'consciousness_scaffold', 'cognitive_scaffold', 'symbolic_scaffold',  # Scaffolds
            'cognitive_stage', 'smart_specific', 'smart_measurable',  # Cognitive + SMART
            'schema_version', 'definition_version'  # Metadata
        ]
        
        neural_score = sum(1 for key in neural_keys if key in architecture)
        agent_score = sum(1 for key in agent_keys if key in architecture)
        voxsigil_score = sum(1 for key in voxsigil_keys if key in architecture)
        enhanced_genome_score = sum(1 for key in enhanced_genome_keys if key in architecture)
        
        # If it has enhanced genome structure (3+ keys), treat as agent
        if enhanced_genome_score >= 3:
            return False
        
        # If it has Voxsigil structure, it's definitely an agent, not a neural architecture
        if voxsigil_score >= 3:  # Has at least 3 Voxsigil keys
            return False
            
        return neural_score >= agent_score and neural_score > 0
    
    def _validate_neural_architecture(self, architecture: Dict[str, Any], 
                                    target_schema: str) -> Dict[str, Any]:
        """Validate a raw neural architecture with schema-inspired principles"""
        
        validation_result = {
            "valid": False,
            "missing_required": [],
            "validation_errors": [],
            "compliance_score": 0.0,
            "suggestions": []
        }
        
        compliance_factors = []
        
        # Layer complexity (architectural scaffolding principle)
        if 'layers' in architecture:
            try:
                layer_count = len(architecture['layers']) if hasattr(architecture['layers'], '__len__') else int(architecture['layers'])
                if layer_count >= 3:  # Multi-layer complexity
                    compliance_factors.append(0.2)
                elif layer_count >= 2:
                    compliance_factors.append(0.1)
            except (ValueError, TypeError):
                compliance_factors.append(0.05)  # Minimal score for invalid layers
                
        # Input-output dimensionality (sensory-response principle)
        if 'input_dim' in architecture and 'output_dim' in architecture:
            try:
                input_dim = float(architecture['input_dim'])
                output_dim = float(architecture['output_dim'])
                if input_dim > 1 and output_dim > 1:  # Multi-dimensional processing
                    compliance_factors.append(0.2)
                else:
                    compliance_factors.append(0.1)
            except (ValueError, TypeError):
                compliance_factors.append(0.05)  # Minimal score for invalid dimensions
                
        # Activation diversity (cognitive modeling principle)
        if 'activations' in architecture:
            try:
                activations = architecture['activations']
                if hasattr(activations, '__iter__') and not isinstance(activations, str):
                    unique_activations = len(set(activations))
                elif isinstance(activations, str):
                    unique_activations = 1
                else:
                    unique_activations = 1
                    
                if unique_activations >= 2:  # Diverse processing modes
                    compliance_factors.append(0.2)
                elif unique_activations >= 1:
                    compliance_factors.append(0.1)
            except (ValueError, TypeError):
                compliance_factors.append(0.05)  # Minimal score for invalid activations
                
        # Device optimization (practical application principle)
        if 'device' in architecture:
            if architecture['device'] == 'cuda':  # Optimized processing
                compliance_factors.append(0.2)
            else:
                compliance_factors.append(0.1)
                
        # Schema constraint alignment
        if hasattr(self, 'constraint_library') and self.constraint_library:
            # Check if architecture aligns with loaded constraints
            constraint_alignment = min(len(compliance_factors) / 4.0, 1.0)
            compliance_factors.append(constraint_alignment * 0.2)
            
        # Calculate final compliance score
        validation_result["compliance_score"] = float(sum(compliance_factors))
        validation_result["valid"] = float(validation_result["compliance_score"]) >= 0.6
        
        logger.info(f"Validation complete: {validation_result['compliance_score']:.2f} compliance")
        return validation_result
    
    def _validate_agent_architecture(self, architecture: Dict[str, Any], 
                                   target_schema: str) -> Dict[str, Any]:
        """Original validation logic for complete agent specifications"""
        
        validation_result = {
            "valid": False,
            "missing_required": [],
            "validation_errors": [],
            "compliance_score": 0.0,
            "suggestions": []
        }
        
        # Get relevant constraints
        constraints = []
        if target_schema == "voxsigil_1_8_omega" and "omega_complete_agent" in self.constraint_library:
            constraints.append(self.constraint_library["omega_complete_agent"])
        elif target_schema == "voxsigil_1_5" and "v15_cognitive_modeling" in self.constraint_library:
            constraints.append(self.constraint_library["v15_cognitive_modeling"])
        elif target_schema == "voxsigil_1_4_uni":
            # For voxsigil_1_4_uni, use available constraints or create basic validation
            if "omega_complete_agent" in self.constraint_library:
                constraints.append(self.constraint_library["omega_complete_agent"])
            elif "v15_cognitive_modeling" in self.constraint_library:
                constraints.append(self.constraint_library["v15_cognitive_modeling"])
        
        logger.info(f"Agent validation for {target_schema}: found {len(constraints)} constraints")
        logger.info(f"Architecture keys: {list(architecture.keys())[:5]}...")  # Show first 5 keys
        
        # Enhanced genome validation (for ArchitectureGenome with VoxSIGIL fields)
        if self._is_enhanced_genome(architecture):
            return self._validate_enhanced_genome(architecture, target_schema, validation_result)
        
        # If no specific constraints found, provide basic validation for any generated agent
        if not constraints and len(architecture) > 0:
            logger.info(f"No specific constraints for {target_schema}, using basic validation")
            validation_result["compliance_score"] = 0.7  # Basic passing score
            validation_result["valid"] = True
            validation_result["suggestions"].append(f"Used basic validation for schema {target_schema}")
            return validation_result
            
        # Validate against each constraint
        total_requirements = 0
        met_requirements = 0
        
        for i, constraint in enumerate(constraints):
            logger.info(f"Constraint {i+1}: {constraint.constraint_type} requires {len(constraint.required_fields)} fields")
            total_requirements += len(constraint.required_fields)
            
            for required_field in constraint.required_fields:
                if required_field in architecture:
                    met_requirements += 1
                else:
                    validation_result["missing_required"].append(required_field)
            
            # Check template requirements if available
            if hasattr(constraint, 'templates') and constraint.templates:
                for template_key, template_value in constraint.templates.items():
                    if template_key in architecture:
                        met_requirements += 0.5  # Partial credit for template matching
                    
        # Calculate compliance score
        if total_requirements > 0:
            validation_result["compliance_score"] = float(met_requirements) / float(total_requirements)
            validation_result["valid"] = float(validation_result["compliance_score"]) >= 0.8
            
        logger.info(f"Validation complete: {validation_result['compliance_score']:.2f} compliance")
        return validation_result
        
    def generate_agent_from_evolution(self, evolved_params: Dict[str, Any], 
                                    base_schema: str = "voxsigil_1_8_omega") -> Dict[str, Any]:
        """Generate a complete Voxsigil agent definition from evolved parameters"""
        
        # Start with a basic structure that includes required fields
        agent_definition = {
            "sigil": f"EVOLVED_AGENT_{hash(str(evolved_params)) % 100000}",
            "name": evolved_params.get("description", "Evolved Agent"),
            "principle": "Schema-driven evolved cognitive agent",
            "usage": "Autonomous cognitive processing and decision making",
            "SMART_MRAP": {
                "Specific": "Designed for specific cognitive tasks",
                "Measurable": "Performance trackable through validation metrics", 
                "Achievable": "Built within schema constraints",
                "Relevant": "Aligned with evolution parameters",
                "Transferable": "Adaptable to various contexts"
            },
            "metadata": {
                "schema_version": base_schema,
                "generation_method": "schema_driven_evolution",
                "created_timestamp": datetime.now().isoformat(),
                "source_constraints": list(self.constraint_library.keys()),
                "evolution_parameters": evolved_params
            }
        }
        
        # Apply evolved parameters (overlay on top of base structure)
        agent_definition.update(evolved_params)
                
        logger.info("Generated complete Voxsigil agent from evolved parameters")
        return agent_definition
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the schema-driven evolution system"""
        
        return {
            "schemas_loaded": len(self.schemas),
            "constraint_types": len(self.constraint_library),
            "available_schemas": list(self.schemas.keys()),
            "constraint_categories": list(self.constraint_library.keys()),
            "integration_status": "active",
            "last_update": datetime.now().isoformat()
        }

    def _is_enhanced_genome(self, architecture: Dict[str, Any]) -> bool:
        """Check if this is an enhanced ArchitectureGenome with VoxSIGIL fields"""
        enhanced_indicators = [
            'sigil', 'name', 'principle',  # Core identity
            'consciousness_scaffold', 'cognitive_scaffold', 'symbolic_scaffold',  # Scaffolds
            'cognitive_stage', 'smart_specific', 'smart_measurable',  # Cognitive/SMART
            'schema_version', 'definition_version'  # Metadata
        ]
        score = sum(1 for key in enhanced_indicators if key in architecture)
        return score >= 5  # If has 5+ enhanced fields, it's our genome
    
    def _validate_enhanced_genome(self, architecture: Dict[str, Any], 
                                 target_schema: str,
                                 validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Custom validation for enhanced ArchitectureGenome"""
        
        # Category scoring
        category_scores = {}
        
        # Core VoxSIGIL fields (0.15 weight)
        core_fields = ['sigil', 'name', 'principle']
        core_present = sum(1 for f in core_fields if f in architecture)
        category_scores['core'] = (core_present / len(core_fields)) * 0.15
        
        # Scaffold fields (0.25 weight)
        scaffold_fields = ['consciousness_scaffold', 'cognitive_scaffold', 'symbolic_scaffold',
                          'consciousness_scaffold_level', 'cognitive_scaffold_role', 
                          'symbolic_orchestration_contribution']
        scaffold_present = sum(1 for f in scaffold_fields if f in architecture)
        category_scores['scaffolds'] = (scaffold_present / len(scaffold_fields)) * 0.25
        
        # SMART_MRAP fields (0.15 weight)
        smart_fields = ['smart_specific', 'smart_measurable', 'smart_achievable',
                       'smart_relevant', 'smart_transferable']
        smart_present = sum(1 for f in smart_fields if f in architecture)
        category_scores['smart'] = (smart_present / len(smart_fields)) * 0.15
        
        # Cognitive modeling fields (0.15 weight) - Using actual field names
        cognitive_fields = ['cognitive_stage', 'developmental_model', 'solo_taxonomy_level',
                           'strategic_goals', 'goal_alignment_strength']
        cognitive_present = sum(1 for f in cognitive_fields if f in architecture)
        category_scores['cognitive'] = (cognitive_present / len(cognitive_fields)) * 0.15
        
        # Learning architecture fields (0.15 weight) - Using actual field names
        learning_fields = ['primary_learning_paradigm', 'continual_learning_enabled', 
                          'catastrophic_forgetting_mitigation', 'memory_architecture_type',
                          'memory_consolidation_enabled']
        learning_present = sum(1 for f in learning_fields if f in architecture)
        category_scores['learning'] = (learning_present / len(learning_fields)) * 0.15
        
        # Metacognition fields (0.15 weight) - Using actual field names
        metacog_fields = ['has_self_model', 'reflective_inference_enabled', 'introspection_capability',
                         'self_correction_enabled', 'hallucination_detection_enabled']
        metacog_present = sum(1 for f in metacog_fields if f in architecture)
        category_scores['metacognition'] = (metacog_present / len(metacog_fields)) * 0.15
        
        # Calculate total compliance score
        total_score = sum(category_scores.values())
        
        validation_result["compliance_score"] = total_score
        validation_result["valid"] = total_score >= 0.5
        validation_result["category_scores"] = category_scores
        
        # Add category-specific feedback
        for category, score in category_scores.items():
            weight = 0.15 if category in ['core', 'smart', 'cognitive', 'learning', 'metacognition'] else 0.25
            if score < weight * 0.5:  # Less than half the category weight
                validation_result["suggestions"].append(
                    f"Consider adding more {category} fields to improve compliance"
                )
        
        logger.info(f"Enhanced genome validation: {total_score:.3f} compliance")
        logger.info(f"Category breakdown: {category_scores}")
        
        return validation_result

# Global instance for system-wide access
schema_evolution_integrator = None

def get_schema_integrator(schema_directory: str = "training/schema") -> SchemaEvolutionIntegrator:
    """Get or create the global schema evolution integrator"""
    global schema_evolution_integrator
    
    if schema_evolution_integrator is None:
        schema_evolution_integrator = SchemaEvolutionIntegrator(schema_directory)
        
    return schema_evolution_integrator

# Integration helper functions for EVO/NAS systems
def get_evolution_constraints(target: str) -> Dict[str, Any]:
    """Helper function for EVO systems to get schema constraints"""
    integrator = get_schema_integrator()
    return integrator.get_evo_constraints(target)

def get_nas_templates(target: str) -> Dict[str, Any]:
    """Helper function for NAS systems to get architecture templates"""
    integrator = get_schema_integrator()
    return integrator.get_nas_constraints(target)

def validate_architecture(arch: Dict[str, Any], schema: str = "voxsigil_1_8_omega") -> Dict[str, Any]:
    """Helper function to validate evolved architectures"""
    integrator = get_schema_integrator()
    return integrator.validate_evolved_architecture(arch, schema)

def create_voxsigil_agent(evolved_params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to create a Voxsigil agent from evolution results"""
    integrator = get_schema_integrator()
    return integrator.generate_agent_from_evolution(evolved_params)

if __name__ == "__main__":
    # Test the schema integration system
    logging.basicConfig(level=logging.INFO)
    
    integrator = SchemaEvolutionIntegrator()
    status = integrator.get_system_status()
    
    print("🧬 Schema-Driven Evolution System Status:")
    for key, value in status.items():
        print(f"  - {key}: {value}")
        
    # Test constraint generation
    evo_constraints = integrator.get_evo_constraints("test_agent")
    nas_constraints = integrator.get_nas_constraints("test_architecture")
    
    print(f"\n🧬 EVO constraints generated: {len(evo_constraints)} categories")
    print(f"[NAS] NAS constraints generated: {len(nas_constraints)} categories")